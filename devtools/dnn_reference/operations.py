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

from typing import Any, Iterable

from .interfaces import DnnReferenceOperation, PreparedOperation


class DnnOperationNotImplementedError(RuntimeError):
    """Raised when a provider has no reference implementation for an op."""


class OperationRegistry:
    """Small per-provider registry of independently implemented DNN ops."""

    def __init__(
        self, operations: Iterable[DnnReferenceOperation] = ()
    ) -> None:
        self._operations: dict[str, DnnReferenceOperation] = {}
        for operation in operations:
            self.register(operation)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._operations)

    def register(
        self, operation: DnnReferenceOperation
    ) -> DnnReferenceOperation:
        name = getattr(operation, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("DNN reference operation name must be non-empty")
        for method_name in ("supports_dtype", "run", "prepare"):
            if not callable(getattr(operation, method_name, None)):
                raise TypeError(
                    f"DNN reference operation {name!r} is missing callable "
                    f"method: {method_name}"
                )
        if name in self._operations:
            raise ValueError(
                f"DNN reference operation is already registered: {name}"
            )
        self._operations[name] = operation
        return operation

    def require(
        self, op_name: str, *, vendor_name: str
    ) -> DnnReferenceOperation:
        try:
            return self._operations[op_name]
        except KeyError as exc:
            raise DnnOperationNotImplementedError(
                f"DNN reference operation {op_name!r} is not implemented "
                f"for vendor: {vendor_name}"
            ) from exc


class RegisteredOperationProvider:
    """Mixin implementing the common provider API through an op registry."""

    vendor_name: str

    def _set_operations(
        self, operations: Iterable[DnnReferenceOperation]
    ) -> None:
        self._operation_registry = OperationRegistry(operations)

    @property
    def operation_names(self) -> tuple[str, ...]:
        return self._operation_registry.names

    def get_operation(self, op_name: str) -> DnnReferenceOperation:
        return self._operation_registry.require(
            op_name, vendor_name=self.vendor_name
        )

    def supports(self, op_name: str, dtype: Any) -> bool:
        try:
            operation = self.get_operation(op_name)
        except DnnOperationNotImplementedError:
            return False
        return bool(operation.supports_dtype(dtype))

    def supports_dtype(self, dtype: Any) -> bool:
        """Compatibility query; new code should call supports(op, dtype)."""
        return any(
            self.get_operation(name).supports_dtype(dtype)
            for name in self.operation_names
        )

    def run(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get_operation(op_name).run(*args, **kwargs)

    def prepare(
        self, op_name: str, *args: Any, **kwargs: Any
    ) -> PreparedOperation:
        return self.get_operation(op_name).prepare(*args, **kwargs)
