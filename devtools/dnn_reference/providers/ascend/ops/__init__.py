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

"""Independently registered ACLNN reference operations."""

from .abs import AscendAbsOperation, configure_abs
from .add import AscendAddOperation, configure_add
from .binary import AscendBinaryOperation, create_binary_operations
from .binary_select import AscendBinarySelectOperation
from .sigmoid_backward import AscendSigmoidBackwardOperation
from .norm import (
    AscendBatchNormInferenceOperation,
    AscendBatchNormOperation,
    AscendLayerNormOperation,
    AscendRmsNormOperation,
    create_norm_operations,
)
from .reduction import AscendReductionOperation
from .unary import AscendUnaryOperation, create_unary_operations
from .utility import (
    AscendReshapeOperation,
    AscendUtilityOperation,
    create_utility_operations,
)

__all__ = (
    "AscendAbsOperation",
    "AscendAddOperation",
    "AscendBinaryOperation",
    "AscendBinarySelectOperation",
    "AscendSigmoidBackwardOperation",
    "AscendBatchNormInferenceOperation",
    "AscendBatchNormOperation",
    "AscendLayerNormOperation",
    "AscendRmsNormOperation",
    "AscendReductionOperation",
    "AscendUnaryOperation",
    "AscendReshapeOperation",
    "AscendUtilityOperation",
    "configure_abs",
    "configure_add",
    "create_binary_operations",
    "create_norm_operations",
    "create_unary_operations",
    "create_utility_operations",
)
