"""Shared validation for portable extended operators on CoreX."""

from .common import FLOAT_TYPES as FLOAT_DATA_TYPES
from .common import POINTER_TYPES as TRITON_POINTER_TYPES
from .common import _require_integer
from .graph_tensor import _has_non_overlapping_strides

from .tensor import _is_row_major_contiguous
