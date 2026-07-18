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

import torch

from flag_dnn.ops.batch_norm import batchnorm_forward


def batchnorm(
    input: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    in_running_mean: torch.Tensor,
    in_running_var: torch.Tensor,
    epsilon,
    momentum,
    peer_stats=None,
    *,
    compute_data_type=None,
    name: str = "",
):
    del compute_data_type, name
    return batchnorm_forward(
        input,
        scale,
        bias,
        in_running_mean,
        in_running_var,
        epsilon,
        momentum,
        peer_stats=peer_stats,
    )
