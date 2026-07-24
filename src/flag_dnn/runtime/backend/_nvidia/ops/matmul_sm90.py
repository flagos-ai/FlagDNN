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
    family: Literal[
        "lowp",
        "lowp_cublaslt",
        "lowp_persistent",
        "tf32",
        "fp8",
        "fp8_tma",
        "tf32_cublaslt",
        "tf32_small",
    ]
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
    Sm90MatmulKey(
        4,
        4096,
        4096,
        4096,
        torch.float16,
        torch.float16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, False),
    Sm90MatmulKey(
        4,
        4096,
        4096,
        4096,
        torch.bfloat16,
        torch.bfloat16,
        "float32",
    ): Sm90MatmulConfig("lowp", 128, 256, 64, 3, 8, 1, 168, False),
}

_LOWP_PERSISTENT_CONFIG = Sm90MatmulConfig(
    "lowp_persistent", 128, 256, 64, 4, 8, 1, 232, False
)

_LOWP_CUBLASLT_CONFIG = Sm90MatmulConfig(
    "lowp_cublaslt", 1, 1, 1, 1, 1, 1, 0, False
)
for _input_dtype in (torch.float16, torch.bfloat16):
    _VALIDATED_CONFIGS[
        Sm90MatmulKey(
            32,
            512,
            512,
            512,
            _input_dtype,
            _input_dtype,
            "float32",
        )
    ] = _LOWP_PERSISTENT_CONFIG

for _batch, _m, _n, _k in (
    (32, 512, 512, 512),
    (16, 2048, 2048, 512),
):
    _VALIDATED_CONFIGS[
        Sm90MatmulKey(
            _batch,
            _m,
            _n,
            _k,
            torch.bfloat16,
            torch.bfloat16,
            "float32",
        )
    ] = _LOWP_CUBLASLT_CONFIG

_FP8_TMA_CONFIG = Sm90MatmulConfig("fp8_tma", 256, 64, 128, 3, 8, 1, 0, False)
_FP8_TMA_SHAPES = (
    (32, 512, 512, 512),
    (16, 1024, 1024, 1024),
    (8, 2048, 2048, 2048),
    (4, 4096, 4096, 4096),
    (16, 2048, 2048, 512),
    (32, 1024, 1024, 4096),
)
for _input_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
    for _batch, _m, _n, _k in _FP8_TMA_SHAPES:
        _VALIDATED_CONFIGS[
            Sm90MatmulKey(
                _batch,
                _m,
                _n,
                _k,
                _input_dtype,
                torch.float32,
                "fast_float_for_fp8",
            )
        ] = _FP8_TMA_CONFIG

_TF32_CUBLASLT_CONFIG = Sm90MatmulConfig(
    "tf32_cublaslt", 1, 1, 1, 1, 1, 1, 0, False
)
for _batch, _m, _n, _k in _FP8_TMA_SHAPES:
    _VALIDATED_CONFIGS[
        Sm90MatmulKey(
            _batch,
            _m,
            _n,
            _k,
            torch.float32,
            torch.float32,
            "tf32",
        )
    ] = _TF32_CUBLASLT_CONFIG

_TF32_SMALL_CONFIG = Sm90MatmulConfig("tf32_small", 1, 1, 1, 1, 1, 1, 0, False)
for _batch, _m, _n, _k in (
    (4, 16, 24, 32),
    (8, 32, 32, 64),
):
    _VALIDATED_CONFIGS[
        Sm90MatmulKey(
            _batch,
            _m,
            _n,
            _k,
            torch.float32,
            torch.float32,
            "tf32",
        )
    ] = _TF32_SMALL_CONFIG


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
    if config.family == "lowp_persistent":
        from .matmul_persistent_sm90 import (
            run_persistent_sm90_matmul,
        )

        run_persistent_sm90_matmul(a, b, c, config=config)
        return True
    if config.family == "lowp_cublaslt":
        from flag_dnn.runtime.backend._nvidia.ops.matmul_cublaslt import (
            run_cublaslt_bf16_matmul,
        )

        run_cublaslt_bf16_matmul(a, b, c)
        return True
    if config.family == "tf32_small":
        from flag_dnn.runtime.backend._nvidia.ops.matmul_small_sm90 import (
            run_small_tf32_matmul,
        )

        run_small_tf32_matmul(a, b, c)
        return True
    if config.family == "tf32_cublaslt":
        from flag_dnn.runtime.backend._nvidia.ops.matmul_cublaslt import (
            run_cublaslt_tf32_matmul,
        )

        run_cublaslt_tf32_matmul(a, b, c)
        return True
    if config.family == "fp8_tma":
        from flag_dnn.runtime.backend._nvidia.ops.matmul_fp8_sm90 import (
            run_sm90_fp8_matmul,
        )

        run_sm90_fp8_matmul(a, b, c, config=config)
        return True
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90_gluon import (
        run_sm90_matmul,
    )

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
    if config.family == "lowp_persistent":
        from .matmul_persistent_sm90 import (
            prepare_persistent_sm90_matmul_dynamic_output,
        )

        launcher = prepare_persistent_sm90_matmul_dynamic_output(
            a,
            b,
            output_dtype=c.dtype,
            config=config,
        )
        return lambda: launcher(c)
    if config.family == "lowp_cublaslt":
        from flag_dnn.runtime.backend._nvidia.ops.matmul_cublaslt import (
            prepare_cublaslt_bf16_matmul_dynamic_output,
        )

        launcher = prepare_cublaslt_bf16_matmul_dynamic_output(
            a,
            b,
            output_dtype=c.dtype,
        )
        return lambda: launcher(c)
    if config.family == "tf32_small":
        from flag_dnn.runtime.backend._nvidia.ops.matmul_small_sm90 import (
            prepare_small_tf32_matmul_dynamic_output,
        )

        launcher = prepare_small_tf32_matmul_dynamic_output(
            a,
            b,
            output_dtype=c.dtype,
        )
        return lambda: launcher(c)
    if config.family == "tf32_cublaslt":
        from flag_dnn.runtime.backend._nvidia.ops.matmul_cublaslt import (
            prepare_cublaslt_tf32_matmul_dynamic_output,
        )

        launcher = prepare_cublaslt_tf32_matmul_dynamic_output(
            a,
            b,
            output_dtype=c.dtype,
        )
        return lambda: launcher(c)
    if config.family == "fp8_tma":
        from flag_dnn.runtime.backend._nvidia.ops.matmul_fp8_sm90 import (
            prepare_sm90_fp8_matmul_dynamic_output,
        )

        launcher = prepare_sm90_fp8_matmul_dynamic_output(
            a,
            b,
            output_dtype=c.dtype,
            config=config,
        )
        return lambda: launcher(c)
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90_gluon import (
        prepare_sm90_matmul,
    )

    return prepare_sm90_matmul(a, b, c, config=config)
