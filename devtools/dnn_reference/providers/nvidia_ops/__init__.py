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

"""Independently registered cuDNN reference operations."""

from .abs import NvidiaAbsOperation
from .add import NvidiaAddOperation
from .binary import NvidiaBinaryOperation, create_binary_operations
from .binary_select import NvidiaBinarySelectOperation
from .batchnorm_inference import NvidiaBatchNormInferenceOperation
from .causal_conv1d import NvidiaCausalConv1dOperation
from .rmsnorm_rht_amax import NvidiaRmsNormRhtAmaxOperation
from .norm import (
    NvidiaBatchNormOperation,
    NvidiaLayerNormOperation,
    NvidiaRmsNormOperation,
    create_norm_operations,
)
from .sigmoid_backward import NvidiaSigmoidBackwardOperation
from .reduction import NvidiaReductionOperation
from .common import NvidiaContext
from .unary import NvidiaUnaryOperation, create_unary_operations
from .utility import NvidiaUtilityOperation, create_utility_operations

__all__ = (
    "NvidiaAbsOperation",
    "NvidiaAddOperation",
    "NvidiaBinaryOperation",
    "NvidiaBinarySelectOperation",
    "NvidiaBatchNormInferenceOperation",
    "NvidiaBatchNormOperation",
    "NvidiaCausalConv1dOperation",
    "NvidiaLayerNormOperation",
    "NvidiaRmsNormOperation",
    "NvidiaRmsNormRhtAmaxOperation",
    "NvidiaSigmoidBackwardOperation",
    "NvidiaReductionOperation",
    "NvidiaContext",
    "NvidiaUnaryOperation",
    "NvidiaUtilityOperation",
    "create_binary_operations",
    "create_norm_operations",
    "create_unary_operations",
    "create_utility_operations",
)
