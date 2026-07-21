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

"""Compatibility aliases for the shared DNN reference provider."""

from typing import Optional, Tuple

from devtools.dnn_reference import (
    DnnProvider as DnnOracle,
    DnnProviderNotImplementedError,
    create_provider,
)

from tests import consts


Shape = Tuple[int, ...]
AddCase = Tuple[Shape, Shape]
OracleNotImplementedError = DnnProviderNotImplementedError


def get_add_test_cases(
    vendor_name: Optional[str] = None,
) -> Tuple[AddCase, ...]:
    return consts.get_add_test_cases(vendor_name)


def create_oracle(vendor_name: Optional[str] = None) -> DnnOracle:
    return create_provider(vendor_name)
