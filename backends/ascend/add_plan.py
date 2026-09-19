"""Compatibility imports; Ascend graph lowering lives in dispatch."""

from .dispatch.common import (
    BINARY_POINTWISE_MODES,
    BOOLEAN_COMPUTE_DATA_TYPE,
    COMPARISON_POINTWISE_OPERATIONS,
    ELEMENT_SIZES,
    FLOAT32_MAX,
    GRAPH_SCHEMA_VERSION,
    GRAPH_WORKSPACE_ALIGNMENT,
    LOGICAL_POINTWISE_OPERATIONS,
    MAX_GRAPH_NODES,
    MAX_I32,
    MAX_I64,
    MAX_RANK,
    NUMERIC_COMPUTE_DATA_TYPE,
    NUMERIC_STORAGE_DATA_TYPES,
    POINTWISE_OPERATIONS,
    PointwiseGraphPlan,
    PointwiseStagePlan,
    REDUCTION_MODES,
    SUPPORTED_BATCHNORM_OPERATIONS,
    SUPPORTED_BINARY_OPERATIONS,
    SUPPORTED_CONVOLUTION_OPERATIONS,
    SUPPORTED_LAYERNORM_OPERATIONS,
    SUPPORTED_LAYOUT_OPERATIONS,
    SUPPORTED_MATMUL_OPERATIONS,
    SUPPORTED_OPERATIONS,
    SUPPORTED_POINTWISE_OPERATIONS,
    SUPPORTED_REDUCTION_OPERATIONS,
    SUPPORTED_RMSNORM_OPERATIONS,
    SUPPORTED_TERNARY_OPERATIONS,
    SUPPORTED_UNARY_OPERATIONS,
    TERNARY_POINTWISE_MODES,
    TensorPlan,
    UNARY_POINTWISE_MODES,
    require_f32,
    require_integer,
    require_integer_list,
    require_list,
    require_object,
)
from .dispatch.convolution import (
    _convolution_fprop_meta,
)
from .dispatch.graph import (
    _align_up,
    _argument_source,
    _stage_workspace,
    _workspace_layout,
    plan_graph,
    plan_binary_graph,
    plan_pointwise_graph,
)
from .dispatch.layout import (
    _is_dense_permutation_layout,
    _is_row_major_layout,
    _layout_meta,
    _layout_mode,
)
from .dispatch.matmul import (
    _matmul_meta,
)
from .dispatch.normalization import (
    _batchnorm_inference_meta,
    _batchnorm_training_meta,
)
from .dispatch.pointwise import (
    _broadcast_dimensions,
    _can_use_contiguous_kernel,
    _can_use_ternary_contiguous_kernel,
    _can_use_unary_contiguous_kernel,
    _optional_equal_f32,
    _optional_mode,
    _parse_unary_attributes,
    _strided_meta,
    _ternary_strided_meta,
    _unary_strided_meta,
)
from .dispatch.reduction import (
    _reduction_meta,
)
from .dispatch.tensor import (
    _effective_strides,
    _has_non_overlapping_strides,
    _is_physically_dense,
    _is_row_major_tensor,
    _parse_named_ports,
    _parse_port,
    _parse_tensor_table,
    _storage_elements,
)
