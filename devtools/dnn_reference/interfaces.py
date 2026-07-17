from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

if TYPE_CHECKING:
    import torch


Number = Union[int, float]


class PreparedOperation(Protocol):
    output: torch.Tensor

    def run(self) -> torch.Tensor: ...

    def close(self) -> None: ...


class DnnProvider(Protocol):
    vendor_name: str
    implementation: str
    display_name: str

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> torch.Tensor: ...

    def abs(self, x: torch.Tensor) -> torch.Tensor: ...

    def prepare_add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> PreparedOperation: ...

    def supports_dtype(self, dtype: torch.dtype) -> bool: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...
