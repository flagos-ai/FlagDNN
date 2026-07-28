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

"""Private hooks connecting portable orchestration to Ascend kernels."""

from .ops.binary import launch_dense_binary, prepare_dense_binary
from .ops.batchnorm import (
    prepare_dense_batchnorm_inference,
    prepare_dense_batchnorm_training,
)
from .ops.causal_conv1d import prepare_causal_conv1d
from .ops.conv import prepare_conv
from .ops.matmul import matmul_3d_out, prepare_matmul
from .ops.norm import prepare_dense_layernorm, prepare_dense_rmsnorm
from .ops.pairwise import (
    prepare_dense_add_square,
    prepare_dense_pow,
    prepare_dense_sigmoid_backward,
)
from .ops.reduction import prepare_dense_reduction
from .ops.sdpa import prepare_sdpa
from .ops.sdpa_backward import prepare_sdpa_backward
from .ops.unary import prepare_dense_unary
from .ops.utility import (
    prepare_dense_concatenate,
    prepare_dense_gen_index,
)

__all__ = (
    "launch_dense_binary",
    "matmul_3d_out",
    "prepare_matmul",
    "prepare_sdpa",
    "prepare_sdpa_backward",
    "prepare_causal_conv1d",
    "prepare_conv",
    "prepare_dense_batchnorm_inference",
    "prepare_dense_batchnorm_training",
    "prepare_dense_add_square",
    "prepare_dense_binary",
    "prepare_dense_concatenate",
    "prepare_dense_gen_index",
    "prepare_dense_layernorm",
    "prepare_dense_pow",
    "prepare_dense_reduction",
    "prepare_dense_rmsnorm",
    "prepare_dense_sigmoid_backward",
    "prepare_dense_unary",
)
