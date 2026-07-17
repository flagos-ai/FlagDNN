"""Compatibility imports for the shared ACLNN wrapper builder."""

from devtools.dnn_reference.providers.ascend.build import (
    CannLayout,
    build_aclnn_oracle,
    find_cann_layout,
)

__all__: list[str] = [
    "CannLayout",
    "build_aclnn_oracle",
    "find_cann_layout",
]
