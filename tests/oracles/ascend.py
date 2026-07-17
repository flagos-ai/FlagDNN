"""Compatibility adapter for the shared Ascend DNN provider."""

from devtools.dnn_reference.providers.ascend.provider import (
    AscendDnnOracle,
    create_oracle,
)

__all__: list[str] = ["AscendDnnOracle", "create_oracle"]
