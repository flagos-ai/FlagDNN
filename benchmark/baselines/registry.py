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

    def supports_dtype(self, dtype) -> bool:
        return self._provider.supports_dtype(dtype)

    def prepare_add(self, x, y, *, alpha=1) -> PreparedOperation:
        return self._provider.prepare_add(x, y, alpha=alpha)

    def close(self) -> None:
        self._provider.close()


def create_baseline(vendor_name: Optional[str] = None) -> DnnBaseline:
    return DnnBaseline(create_provider(vendor_name))
