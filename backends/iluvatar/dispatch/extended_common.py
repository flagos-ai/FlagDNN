"""Shared validation for portable extended operators on CoreX."""

from .common import FLOAT_TYPES as FLOAT_DATA_TYPES  # noqa: F401
from .common import POINTER_TYPES as TRITON_POINTER_TYPES  # noqa: F401
from .common import _require_integer  # noqa: F401
from .graph_tensor import (  # noqa: F401
    _has_non_overlapping_strides,
)

from .tensor import _is_row_major_contiguous  # noqa: F401
