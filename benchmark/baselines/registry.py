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

from typing import Optional

from devtools.dnn_reference import DnnProvider, PreparedOperation
from devtools.dnn_reference import create_provider


class DnnBaseline:
    """Performance-only view of a shared vendor DNN provider."""

    def __init__(self, provider: DnnProvider) -> None:
        self._provider = provider
        self.vendor_name = provider.vendor_name
        self.implementation = provider.implementation
        self.display_name = provider.display_name

    @property
    def operation_names(self) -> tuple[str, ...]:
        return self._provider.operation_names

    def supports(self, op_name, dtype) -> bool:
        return self._provider.supports(op_name, dtype)

    def prepare(self, op_name, *args, **kwargs) -> PreparedOperation:
        return self._provider.prepare(op_name, *args, **kwargs)

    def close(self) -> None:
        self._provider.close()


def create_baseline(vendor_name: Optional[str] = None) -> DnnBaseline:
    return DnnBaseline(create_provider(vendor_name))
