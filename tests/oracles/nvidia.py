"""Compatibility adapter for the shared NVIDIA DNN provider."""

from devtools.dnn_reference.providers.nvidia import (
    NvidiaDnnOracle,
    create_oracle,
)

__all__: list[str] = ["NvidiaDnnOracle", "create_oracle"]
