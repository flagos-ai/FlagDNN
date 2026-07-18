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

from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass(frozen=True)
class Sm90MatmulKey:
    batch: int
    m: int
    n: int
    k: int
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    compute_mode: str


@dataclass(frozen=True)
class Sm90MatmulConfig:
    family: Literal["lowp", "tf32", "fp8"]
    block_m: int
    block_n: int
    block_k: int
    num_buffers: int
    num_warps: int
    grid_multiplier: int
    maxnreg: int
    warp_specialized: bool


_VALIDATED_CONFIGS: dict[Sm90MatmulKey, Sm90MatmulConfig] = {
    Sm90MatmulKey(
        16,
        1024,
        1024,
        1024,
        torch.float16,
        torch.float16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, True),
    Sm90MatmulKey(
        16,
        1024,
        1024,
        1024,
        torch.bfloat16,
        torch.bfloat16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, True),
    Sm90MatmulKey(
        8,
        2048,
        2048,
        2048,
        torch.float16,
        torch.float16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, False),
    Sm90MatmulKey(
        8,
        2048,
        2048,
        2048,
        torch.bfloat16,
        torch.bfloat16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, False),
    Sm90MatmulKey(
        16,
        2048,
        2048,
        512,
        torch.float16,
        torch.float16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, True),
    Sm90MatmulKey(
        16,
        2048,
        2048,
        512,
        torch.bfloat16,
        torch.bfloat16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, True),
    Sm90MatmulKey(
        32,
        1024,
        1024,
        4096,
        torch.float16,
        torch.float16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 2, 168, False),
    Sm90MatmulKey(
        32,
        1024,
        1024,
        4096,
        torch.bfloat16,
        torch.bfloat16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 2, 168, False),
}


def select_sm90_matmul_config(capability, key):
    return _VALIDATED_CONFIGS.get(key) if capability == (9, 0) else None


def launch_sm90_matmul_if_supported(a, b, c, *, compute_mode, capability):
    batch, m, k = map(int, a.shape)
    key = Sm90MatmulKey(
        batch,
        m,
        int(b.shape[2]),
        k,
        a.dtype,
        c.dtype,
        compute_mode,
    )
    config = select_sm90_matmul_config(capability, key)
    if config is None:
        return False
    from flag_dnn.ops.matmul_sm90_gluon import run_sm90_matmul

    run_sm90_matmul(a, b, c, config=config)
    return True


def prepare_sm90_matmul_if_supported(
    a,
    b,
    c,
    *,
    compute_mode,
    capability,
) -> Callable[[], torch.Tensor] | None:
    batch, m, k = map(int, a.shape)
    key = Sm90MatmulKey(
        batch,
        m,
        int(b.shape[2]),
        k,
        a.dtype,
        c.dtype,
        compute_mode,
    )
    config = select_sm90_matmul_config(capability, key)
    if config is None:
        return None
    from flag_dnn.ops.matmul_sm90_gluon import prepare_sm90_matmul

    return prepare_sm90_matmul(a, b, c, config=config)
