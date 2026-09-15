# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon Graph-IR planning for common DNN kernels.

This module is deliberately pure Python and independent of the provider entry
point.  It only selects functions declared for each operation in
``kernels/registry.json``; provider integration may import the operation set,
``parse_node`` and ``plan_kernel_stages`` without importing Triton.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

HYGON_WARP_SIZE = 64
MAX_WORKGROUP_SIZE = 1024
WORKSPACE_ALIGNMENT = 256
DGRAD_EXACT_3X3_S1_WORKSPACE_CAP = 9_437_184
MAX_RANK = 8
MAX_I32 = 2**31 - 1
MIN_I32 = -(2**31)
MAX_U32 = 2**32 - 1
MAX_I64 = 2**63 - 1
MAX_FLOAT32 = 3.4028234663852886e38
UNBOUNDED_DIAGONAL = 1 << 30


# Provider integration must preserve this policy. The installed HCU backend
# defaults tl.dot to IEEE and the convolution entries also hard-code IEEE.
STRICT_DOT_INPUT_PRECISION = "ieee"
ALLOW_TF32 = False
POINTER_TYPES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "fp8_e4m3": "*fp8e4nv",
    "fp8_e5m2": "*fp8e5",
}
ELEMENT_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}
FLOAT_TYPES = frozenset(("float32", "float16", "bfloat16"))
FP8_TYPES = frozenset(("fp8_e4m3", "fp8_e5m2"))
DTYPE_IDS = {"float32": 0, "float16": 1, "bfloat16": 2}

CONVOLUTION_OPERATIONS = frozenset(
    (
        "conv2d_fprop",
        "convolution_fprop",
        "convolution_dgrad",
        "convolution_wgrad",
    )
)
NORMALIZATION_OPERATIONS = frozenset(
    ("layernorm", "rmsnorm", "batchnorm", "batchnorm_inference")
)
ATTENTION_OPERATIONS = frozenset(
    ("sdpa", "sdpa_backward", "sdpa_fp8", "sdpa_fp8_backward")
)
SUPPORTED_OPERATIONS = frozenset(
    (*CONVOLUTION_OPERATIONS, *NORMALIZATION_OPERATIONS, *ATTENTION_OPERATIONS)
)
OPERATIONS = SUPPORTED_OPERATIONS

# Source-of-truth mirror of the registry-selected common or Hygon provider.
# Planning code may not select outside these declared entry points.
_CONV_FPROP_FUNCTIONS = (
    "conv1d_gemm_kernel",
    "conv2d_spatial_nchw_kernel",
    "conv3d_spatial_ncdhw_m_kernel",
    "hygon_conv2d_im2col_nchw_kernel",
    "hygon_conv2d_fprop_stride2_low_ci_nchw_kernel",
    "hygon_conv2d_fprop_standard_3x3_nchw_kernel",
    "hygon_conv2d_fprop_im2col_kernel",
    "hygon_conv2d_fprop_yolo_x_p5_gemm_kernel",
    "hygon_conv2d_fprop_1x1_nchw_kernel",
)
_CONV_DGRAD_FUNCTIONS = (
    "conv_dgrad_nd_kernel",
    "hygon_conv_dgrad2d_1x1_nchw_kernel",
    "hygon_conv_dgrad2d_stride1_kernel",
    "hygon_conv_dgrad2d_stride2_contribution_gemm_kernel",
    "hygon_conv_dgrad2d_stride2_block_pointer_gemm_kernel",
    "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel",
    "hygon_conv_dgrad2d_stride2_parity_kernel",
    "hygon_conv_dgrad2d_exact_3x3_s1_gemm_kernel",
    "hygon_conv_dgrad2d_exact_3x3_s1_col2im_kernel",
)
_CONV_WGRAD_FUNCTIONS = (
    "conv_wgrad_nd_kernel",
    "hygon_conv2d_im2col_nchw_kernel",
    "hygon_conv_wgrad2d_im2row_kernel",
    "hygon_conv_wgrad2d_rowmajor_kernel",
    "hygon_conv_wgrad2d_p5_block_ptr_kernel",
    "hygon_conv_wgrad2d_im2col_kernel",
    "hygon_conv_wgrad2d_im2col_split_kernel",
    "hygon_conv_wgrad2d_stem_split_kernel",
    "hygon_conv_wgrad2d_multirow_split_kernel",
    "hygon_conv_wgrad2d_direct_split_kernel",
    "hygon_conv_wgrad2d_1x1_split_kernel",
    "hygon_conv_wgrad2d_reduce_kernel",
)
REGISTRY_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "conv2d_fprop": _CONV_FPROP_FUNCTIONS,
    "convolution_fprop": _CONV_FPROP_FUNCTIONS,
    "convolution_dgrad": _CONV_DGRAD_FUNCTIONS,
    "convolution_wgrad": _CONV_WGRAD_FUNCTIONS,
    "layernorm": ("layer_norm_kernel",),
    "rmsnorm": ("rms_norm_kernel",),
    "batchnorm": ("batch_norm_nchw_kernel", "batch_norm_kernel"),
    "batchnorm_inference": (
        "batch_norm_inference_nchw_kernel",
        "batch_norm_inference_kernel",
    ),
    "sdpa": ("_sdpa_fwd_kernel",),
    "sdpa_backward": (
        "_zero_contiguous_kernel",
        "_sdpa_bwd_dq_dbias_kernel",
        "_sdpa_bwd_dkdv_kernel",
        "_sdpa_bwd_dk_kernel",
        "_sdpa_bwd_dv_kernel",
    ),
    "sdpa_fp8": (
        "_zero_sdpa_fp8_fwd_amax_kernel",
        "_sdpa_fp8_fwd_kernel",
    ),
    "sdpa_fp8_backward": (
        "_zero_sdpa_fp8_bwd_amax_kernel",
        "_sdpa_fp8_bwd_dq_kernel",
        "_sdpa_fp8_bwd_dkdv_kernel",
    ),
}

ArgumentLayout = tuple[tuple[str, str | int | None], ...]
Grid = tuple[int, int, int]


@dataclass(frozen=True)
class TuningMetadata:
    """Registry-compatible route metadata for an autotuned stage."""

    source: str
    table: str
    key: str
    strategy: str
    warmup: int = 5
    repetitions: int = 10
    meta_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationSchema:
    """Graph IR ports and registry candidates for one DNN operation."""

    family: str
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    functions: tuple[str, ...]
    tuning: TuningMetadata
    conditional_bias: bool = False
    conditional_dbias: bool = False


CONV_FPROP_TUNING = TuningMetadata(
    "convolution.yaml",
    "conv_fprop",
    "n_outputs",
    "convolution",
    meta_keys=(
        "BLOCK_M",
        "BLOCK_HW",
        "BLOCK_OC",
        "BLOCK_CI",
        "BLOCK_K",
        "GROUP_M",
    ),
)
CONV_FPROP_YOLO_X_P5_GEMM_TUNING = TuningMetadata(
    "convolution.yaml",
    "conv_fprop",
    "n_outputs",
    "convolution",
    meta_keys=(
        "BLOCK_M_X",
        "BLOCK_OC_X",
        "BLOCK_K_X",
        "GROUP_M_X",
        "YOLO_X_P5",
    ),
)
CONV_FPROP_STANDARD_3X3_TUNING = TuningMetadata(
    "convolution.yaml",
    "conv_fprop",
    "n_outputs",
    "convolution",
    meta_keys=("BLOCK_OC_S", "BLOCK_K_S"),
)
CONV_DGRAD_TUNING = TuningMetadata(
    "convolution.yaml",
    "conv_dgrad",
    "n_outputs",
    "convolution",
    meta_keys=(
        "BLOCK_M",
        "BLOCK_CI",
        "BLOCK_CO",
        "BLOCK_K",
        "GROUP_M",
        "BLOCK_SIZE",
    ),
)
CONV_DGRAD_EXACT_GEMM_TUNING = TuningMetadata(
    "convolution.yaml",
    "conv_dgrad",
    "n_outputs",
    "convolution",
    meta_keys=("BLOCK_M", "GROUP_M", "EXACT_3X3_S1"),
)
CONV_DGRAD_EXACT_COL2IM_TUNING = TuningMetadata(
    "convolution.yaml",
    "conv_dgrad",
    "n_outputs",
    "convolution",
    meta_keys=("BLOCK_H", "EXACT_3X3_S1"),
)
CONV_WGRAD_TUNING = TuningMetadata(
    "convolution.yaml",
    "conv_wgrad",
    "n_outputs",
    "convolution",
    meta_keys=(
        "BLOCK_M",
        "BLOCK_OC",
        "BLOCK_CI",
        "BLOCK_CI_K",
        "BLOCK_K",
        "BLOCK_SIZE",
    ),
)
CONV_TUNING_BY_OPERATION = {
    "conv2d_fprop": CONV_FPROP_TUNING,
    "convolution_fprop": CONV_FPROP_TUNING,
    "convolution_dgrad": CONV_DGRAD_TUNING,
    "convolution_wgrad": CONV_WGRAD_TUNING,
}
LAYER_NORM_TUNING = TuningMetadata(
    "common.yaml",
    "layer_norm",
    "normalized_elements",
    "fixed_grid",
    meta_keys=("BLOCK_SIZE", "ROWS_PER_PROGRAM"),
)
RMS_NORM_TUNING = TuningMetadata(
    "common.yaml",
    "rms_norm",
    "normalized_elements",
    "fixed_grid",
    meta_keys=("BLOCK_SIZE", "ROWS_PER_PROGRAM"),
)
BATCH_NORM_TUNING = TuningMetadata(
    "normalization.yaml",
    "batch_norm",
    "channels",
    "fixed_grid",
    meta_keys=("BLOCK_SIZE",),
)
BATCH_NORM_INFERENCE_TUNING = TuningMetadata(
    "common.yaml",
    "batch_norm",
    "n_elements",
    "align32",
    meta_keys=("BLOCK_SIZE",),
)
SDPA_TUNING = TuningMetadata(
    "common.yaml",
    "sdpa",
    "sequence_q",
    "attention",
    meta_keys=("BLOCK_M", "BLOCK_N"),
)
SDPA_BACKWARD_TUNING = TuningMetadata(
    "common.yaml",
    "sdpa_backward_dq",
    "sequence_q",
    "attention",
    meta_keys=("BLOCK_M", "BLOCK_N", "BLOCK_D_OUT", "BLOCK_DV_OUT"),
)
SDPA_FP8_TUNING = TuningMetadata(
    "common.yaml",
    "sdpa_fp8",
    "sequence_q",
    "attention",
    meta_keys=("BLOCK_M", "BLOCK_N"),
)
SDPA_FP8_BACKWARD_TUNING = TuningMetadata(
    "common.yaml",
    "sdpa_fp8_backward_dq",
    "sequence_q",
    "attention",
    meta_keys=("BLOCK_M", "BLOCK_N"),
)
INTERNAL_TUNING = TuningMetadata("", "", "", "fixed", meta_keys=("BLOCK",))


def _schema(
    operation: str,
    family: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    tuning: TuningMetadata,
    *,
    conditional_bias: bool = False,
    conditional_dbias: bool = False,
) -> OperationSchema:
    return OperationSchema(
        family,
        inputs,
        outputs,
        REGISTRY_FUNCTIONS[operation],
        tuning,
        conditional_bias,
        conditional_dbias,
    )


OPERATION_SCHEMAS: dict[str, OperationSchema] = {
    "conv2d_fprop": _schema(
        "conv2d_fprop",
        "convolution",
        ("input", "filter"),
        ("output",),
        CONV_FPROP_TUNING,
    ),
    "convolution_fprop": _schema(
        "convolution_fprop",
        "convolution",
        ("input", "filter"),
        ("output",),
        CONV_FPROP_TUNING,
    ),
    "convolution_dgrad": _schema(
        "convolution_dgrad",
        "convolution",
        ("dy", "w"),
        ("dx",),
        CONV_DGRAD_TUNING,
    ),
    "convolution_wgrad": _schema(
        "convolution_wgrad",
        "convolution",
        ("dy", "x"),
        ("dw",),
        CONV_WGRAD_TUNING,
    ),
    "layernorm": _schema(
        "layernorm",
        "normalization",
        ("x", "scale", "bias"),
        ("y", "mean", "inv_variance"),
        LAYER_NORM_TUNING,
    ),
    "rmsnorm": _schema(
        "rmsnorm",
        "normalization",
        ("x", "scale", "bias"),
        ("y", "inv_variance"),
        RMS_NORM_TUNING,
    ),
    "batchnorm": _schema(
        "batchnorm",
        "normalization",
        (
            "x",
            "scale",
            "bias",
            "previous_running_mean",
            "previous_running_variance",
        ),
        (
            "y",
            "mean",
            "inv_variance",
            "next_running_mean",
            "next_running_variance",
        ),
        BATCH_NORM_TUNING,
    ),
    "batchnorm_inference": _schema(
        "batchnorm_inference",
        "normalization",
        ("x", "mean", "inv_variance", "scale", "bias"),
        ("y",),
        BATCH_NORM_INFERENCE_TUNING,
    ),
    "sdpa": _schema(
        "sdpa",
        "attention",
        ("q", "k", "v"),
        ("o", "stats"),
        SDPA_TUNING,
        conditional_bias=True,
    ),
    "sdpa_backward": _schema(
        "sdpa_backward",
        "attention",
        ("q", "k", "v", "o", "do", "stats"),
        ("dq", "dk", "dv"),
        SDPA_BACKWARD_TUNING,
        conditional_bias=True,
        conditional_dbias=True,
    ),
    "sdpa_fp8": _schema(
        "sdpa_fp8",
        "attention",
        (
            "q",
            "k",
            "v",
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_s",
            "scale_s",
            "scale_o",
        ),
        ("o", "stats", "amax_s", "amax_o"),
        SDPA_FP8_TUNING,
    ),
    "sdpa_fp8_backward": _schema(
        "sdpa_fp8_backward",
        "attention",
        (
            "q",
            "k",
            "v",
            "o",
            "do",
            "stats",
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_o",
            "descale_do",
            "descale_s",
            "descale_dp",
            "scale_s",
            "scale_dq",
            "scale_dk",
            "scale_dv",
            "scale_dp",
        ),
        ("dq", "dk", "dv", "amax_dq", "amax_dk", "amax_dv", "amax_dp"),
        SDPA_FP8_BACKWARD_TUNING,
    ),
}


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _attention_full_dimension_block(value: int) -> int:
    """Return a tl.dot-compatible power-of-two block for a full D/V axis."""
    return max(16, _next_power_of_two(value))


def _attention_output_dimension_block(value: int) -> int:
    """Return the bounded output-axis block used by backward attention."""
    return min(32, _attention_full_dimension_block(value))


def _checked_product(values: Sequence[int], name: str) -> int:
    result = math.prod(values)
    if result <= 0 or result > MAX_I64:
        raise ValueError(f"{name} is invalid or overflows int64")
    return result


def _validate_grid(grid: Grid) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_U32
        for value in grid
    ):
        raise ValueError("Hygon launch grid dimensions must fit uint32")


def _validate_launch(num_warps: int, num_stages: int) -> None:
    if (
        isinstance(num_warps, bool)
        or not isinstance(num_warps, int)
        or num_warps <= 0
        or num_warps * HYGON_WARP_SIZE > MAX_WORKGROUP_SIZE
    ):
        raise ValueError(
            "Hygon num_warps must be positive and num_warps * 64 <= 1024"
        )
    if (
        isinstance(num_stages, bool)
        or not isinstance(num_stages, int)
        or not 1 <= num_stages <= 32
    ):
        raise ValueError("Hygon num_stages must be in [1, 32]")


@dataclass(frozen=True)
class GridSpec:
    """A tuning-aware launch formula for one common kernel stage."""

    kind: str
    extents: tuple[int, ...]

    def evaluate(self, constants: Mapping[str, int | float | bool]) -> Grid:
        if self.kind == "fixed":
            result = tuple(self.extents)
        elif self.kind == "linear":
            result = (
                _ceil_div(self.extents[0], _meta_int(constants, "BLOCK_SIZE")),
                1,
                1,
            )
        elif self.kind == "zero":
            result = (
                _ceil_div(self.extents[0], _meta_int(constants, "BLOCK")),
                1,
                1,
            )
        elif self.kind == "norm_rows":
            result = (
                _ceil_div(
                    self.extents[0],
                    _meta_int(constants, "ROWS_PER_PROGRAM"),
                ),
                1,
                1,
            )
        elif self.kind == "batchnorm_channels":
            result = (self.extents[0], 1, 1)
        elif self.kind == "batchnorm_inference_nchw":
            batch, channels, spatial = self.extents
            block = _meta_int(constants, "BLOCK_SIZE")
            block_s = min(_next_power_of_two(spatial), block)
            block_c = max(1, block // block_s)
            result = (
                batch
                * _ceil_div(channels, block_c)
                * _ceil_div(spatial, block_s),
                1,
                1,
            )
        elif self.kind == "conv1d":
            rows, channels, groups = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(channels, _meta_int(constants, "BLOCK_OC")),
                groups,
                1,
            )
        elif self.kind == "conv2d":
            output_hw, channels, batch_groups = self.extents
            result = (
                _ceil_div(output_hw, _meta_int(constants, "BLOCK_HW"))
                * _ceil_div(channels, _meta_int(constants, "BLOCK_OC")),
                batch_groups,
                1,
            )
        elif self.kind == "conv3d":
            rows, channels, groups = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(channels, _meta_int(constants, "BLOCK_OC")),
                groups,
                1,
            )
        elif self.kind == "conv_dgrad":
            rows, channels, groups = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(channels, _meta_int(constants, "BLOCK_CI")),
                groups,
                1,
            )
        elif self.kind == "conv_wgrad":
            out_channels, in_channels, kernel_volume, groups = self.extents
            result = (
                _ceil_div(out_channels, _meta_int(constants, "BLOCK_OC"))
                * _ceil_div(in_channels, _meta_int(constants, "BLOCK_CI")),
                kernel_volume,
                groups,
            )
        elif self.kind == "conv_private_output":
            rows, channels, batches_or_groups = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(channels, _meta_int(constants, "BLOCK_OC")),
                batches_or_groups,
                1,
            )
        elif self.kind == "conv_private_fprop_yolo_x":
            rows, channels, batches = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M_X"))
                * _ceil_div(channels, _meta_int(constants, "BLOCK_OC_X")),
                batches,
                1,
            )
        elif self.kind == "conv_private_fprop_standard_3x3":
            rows, channels, batches = self.extents
            result = (
                rows * _ceil_div(channels, _meta_int(constants, "BLOCK_OC_S")),
                batches,
                1,
            )
        elif self.kind == "conv_private_im2col":
            rows, reduction, batches = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(reduction, _meta_int(constants, "BLOCK_K")),
                batches,
                1,
            )
        elif self.kind == "conv_private_fprop_rows":
            rows, columns, channels, batches = self.extents
            result = (
                rows
                * _ceil_div(columns, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(channels, _meta_int(constants, "BLOCK_OC")),
                batches,
                1,
            )
        elif self.kind == "conv_private_dgrad":
            rows, channels, batches_or_groups = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(channels, _meta_int(constants, "BLOCK_CI")),
                batches_or_groups,
                1,
            )
        elif self.kind == "conv_private_dgrad_columns":
            rows, columns, batches = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(columns, _meta_int(constants, "BLOCK_CI")),
                batches,
                1,
            )
        elif self.kind == "conv_private_dgrad_fixed_2d":
            rows, batch_channels = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_H")),
                batch_channels,
                1,
            )
        elif self.kind == "conv_private_wgrad":
            out_channels, reduction, splits = self.extents
            result = (
                _ceil_div(out_channels, _meta_int(constants, "BLOCK_OC"))
                * _ceil_div(reduction, _meta_int(constants, "BLOCK_CI_K")),
                splits,
                1,
            )
        elif self.kind == "conv_private_wgrad_1x1":
            out_channels, in_channels, split_groups = self.extents
            result = (
                _ceil_div(out_channels, _meta_int(constants, "BLOCK_OC"))
                * _ceil_div(in_channels, _meta_int(constants, "BLOCK_CI")),
                split_groups,
                1,
            )
        elif self.kind in ("sdpa_fwd", "sdpa_fp8_fwd"):
            sequence_q, batch_heads = self.extents
            result = (
                _ceil_div(sequence_q, _meta_int(constants, "BLOCK_M")),
                batch_heads,
                1,
            )
        elif self.kind == "sdpa_dq":
            sequence_q, head_dimension, batch_heads = self.extents
            result = (
                _ceil_div(sequence_q, _meta_int(constants, "BLOCK_M")),
                _ceil_div(head_dimension, _meta_int(constants, "BLOCK_D_OUT")),
                batch_heads,
            )
        elif self.kind == "sdpa_dk":
            sequence_kv, head_dimension, batch_heads = self.extents
            result = (
                _ceil_div(sequence_kv, _meta_int(constants, "BLOCK_N")),
                _ceil_div(head_dimension, _meta_int(constants, "BLOCK_D_OUT")),
                batch_heads,
            )
        elif self.kind == "sdpa_dv":
            sequence_kv, value_dimension, batch_heads = self.extents
            result = (
                _ceil_div(sequence_kv, _meta_int(constants, "BLOCK_N")),
                _ceil_div(
                    value_dimension, _meta_int(constants, "BLOCK_DV_OUT")
                ),
                batch_heads,
            )
        elif self.kind in ("sdpa_fp8_dq", "sdpa_fp8_dkdv"):
            sequence, batch_heads = self.extents
            block_name = "BLOCK_M" if self.kind.endswith("dq") else "BLOCK_N"
            result = (
                _ceil_div(sequence, _meta_int(constants, block_name)),
                batch_heads,
                1,
            )
        else:
            raise ValueError(f"unknown Hygon NN grid kind {self.kind!r}")
        if len(result) != 3:
            raise ValueError("Hygon grid must have three dimensions")
        typed = (int(result[0]), int(result[1]), int(result[2]))
        _validate_grid(typed)
        return typed


@dataclass(frozen=True)
class KernelStagePlan:
    """One registry-declared Triton stage and its complete visible ABI.

    A tensor-layout payload is the index in the parsed node's tensor list. A
    workspace-tensor payload names an entry in NodePlan.workspace_tensors.
    Scalar parameters not declared ``tl.constexpr`` by the registry kernel are
    explicit runtime ABI values. Their layout payload names the entry in
    ``runtime_values``; specialized parameters remain in ``constants``.
    """

    operation: str
    stage_name: str
    function_name: str
    runtime_signature: dict[str, str]
    runtime_values: dict[str, int | float]
    constants: dict[str, int | float | bool]
    default_grid: Grid
    argument_layout: ArgumentLayout
    tuning: TuningMetadata
    tuning_key_value: int
    default_num_warps: int
    default_num_stages: int
    grid_spec: GridSpec
    dependencies: tuple[str, ...] = ()
    precision_mode: str = STRICT_DOT_INPUT_PRECISION
    workspace_alignment: int = WORKSPACE_ALIGNMENT

    @property
    def hidden_argument_layout(self) -> ArgumentLayout:
        return (
            ("global_scratch_pointer", None),
            ("profile_scratch_pointer", None),
        )

    @property
    def full_argument_layout(self) -> ArgumentLayout:
        return self.argument_layout + self.hidden_argument_layout

    def validate_launch(
        self,
        *,
        num_warps: int | None = None,
        num_stages: int | None = None,
    ) -> None:
        _validate_launch(
            self.default_num_warps if num_warps is None else num_warps,
            self.default_num_stages if num_stages is None else num_stages,
        )

    def variant(
        self,
        meta: Mapping[str, object],
        *,
        num_warps: int | None = None,
        num_stages: int | None = None,
    ) -> tuple[dict[str, int | float | bool], Grid]:
        allowed = set(self.tuning.meta_keys).intersection(self.constants)
        unknown = set(meta).difference(allowed)
        if unknown:
            raise ValueError(
                f"{self.stage_name} tuning META contains unsupported keys: "
                + ", ".join(sorted(unknown))
            )
        constants = dict(self.constants)
        for name, value in meta.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"tuning META.{name} must be a positive integer"
                )
            if name.startswith("BLOCK") and value & (value - 1):
                raise ValueError(f"tuning META.{name} must be a power of two")
            constants[name] = value
        self.validate_launch(num_warps=num_warps, num_stages=num_stages)
        return constants, self.grid_spec.evaluate(constants)


@dataclass(frozen=True)
class WorkspaceTensor:
    """Provider-local virtual tensor required between stages."""

    name: str
    data_type: str
    dimensions: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int
    size: int
    alignment: int = WORKSPACE_ALIGNMENT


@dataclass(frozen=True)
class NodePlan:
    """All stages and provider-local workspace for one Graph IR node."""

    operation: str
    stages: tuple[KernelStagePlan, ...]
    workspace_tensors: tuple[WorkspaceTensor, ...] = ()
    workspace_size: int = 0

    def validate_dependencies(self) -> None:
        seen: set[str] = set()
        for stage in self.stages:
            if stage.stage_name in seen:
                raise ValueError(f"duplicate stage name {stage.stage_name!r}")
            missing = set(stage.dependencies).difference(seen)
            if missing:
                raise ValueError(
                    f"stage {stage.stage_name!r} has forward dependencies: "
                    + ", ".join(sorted(missing))
                )
            seen.add(stage.stage_name)


def _meta_int(values: Mapping[str, int | float | bool], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"kernel constant {name} must be a positive integer")
    return value


def _require_object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_integer(
    values: Mapping[str, Any],
    name: str,
    *,
    minimum: int = 1,
    maximum: int = MAX_I64,
) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"parameters.{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(
            f"parameters.{name} must be in [{minimum}, {maximum}]"
        )
    return value


def _require_flag(values: Mapping[str, Any], name: str) -> bool:
    return bool(_require_integer(values, name, minimum=0, maximum=1))


def _require_number(
    values: Mapping[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameters.{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " finite and positive" if positive else " finite"
        raise ValueError(f"parameters.{name} must be{suffix}")
    return result


def _require_integer_list(
    values: Mapping[str, Any],
    name: str,
    length: int,
    *,
    minimum: int,
    maximum: int = MAX_I64,
) -> list[int]:
    raw = _require_sequence(values.get(name), f"parameters.{name}")
    if len(raw) != length:
        raise ValueError(f"parameters.{name} must contain {length} integers")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < minimum
        or item > maximum
        for item in raw
    ):
        raise ValueError(
            f"parameters.{name} values must be integers in "
            f"[{minimum}, {maximum}]"
        )
    return list(raw)


def _has_non_overlapping_strides(
    dimensions: Sequence[int], strides: Sequence[int]
) -> bool:
    axes = sorted(
        (stride, dimension)
        for dimension, stride in zip(dimensions, strides, strict=True)
        if dimension > 1
    )
    required_span = 1
    for stride, dimension in axes:
        if stride < required_span:
            return False
        required_span += (dimension - 1) * stride
        if required_span > MAX_I64:
            return False
    return True


def _is_contiguous(tensor: Mapping[str, Any]) -> bool:
    expected = 1
    for dimension, stride in zip(
        reversed(tensor["dimensions"]), reversed(tensor["strides"])
    ):
        if stride != expected:
            return False
        expected *= dimension
    return True


def tensor_storage_size(tensor: Mapping[str, Any]) -> int:
    dimensions = tensor["dimensions"]
    strides = tensor["strides"]
    elements = 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(dimensions, strides, strict=True)
    )
    size = elements * ELEMENT_SIZES[tensor["data_type"]]
    if size <= 0 or size > MAX_I64:
        raise ValueError("tensor storage size is invalid or overflows int64")
    return size


def _validate_tensor(tensor_value: object, name: str) -> Mapping[str, Any]:
    tensor = _require_object(tensor_value, name)
    uid = tensor.get("uid")
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError(f"{name} UID is invalid")
    data_type = tensor.get("data_type")
    if not isinstance(data_type, str) or data_type not in POINTER_TYPES:
        raise ValueError(f"{name} data type is unsupported: {data_type!r}")
    dimensions = _require_sequence(
        tensor.get("dimensions"), f"{name}.dimensions"
    )
    strides = _require_sequence(tensor.get("strides"), f"{name}.strides")
    if len(dimensions) != len(strides) or len(dimensions) > MAX_RANK:
        raise ValueError(f"{name} rank must be in [0, {MAX_RANK}]")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_I32
        for value in dimensions
    ):
        raise ValueError(f"{name} dimensions must be positive int32 values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_I64
        for value in strides
    ):
        raise ValueError(f"{name} strides must be positive int64 values")
    if not _has_non_overlapping_strides(dimensions, strides):
        raise ValueError(f"{name} strides overlap")
    alignment = tensor.get("alignment", 16)
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, int)
        or alignment <= 0
        or alignment > MAX_I32
        or alignment & (alignment - 1)
    ):
        raise ValueError(f"{name} alignment must be a positive power of two")
    virtual = tensor.get("virtual", False)
    if not isinstance(virtual, bool):
        raise ValueError(f"{name} virtual flag must be boolean")
    tensor_storage_size(tensor)
    return tensor


def _parse_port(
    port_value: object,
    expected_name: str,
    direction: str,
    tensor_registry: Mapping[int, Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]]:
    port = _require_object(port_value, f"node.{direction}")
    if port.get("name") != expected_name:
        raise ValueError(
            f"node {direction} port must be named {expected_name!r}"
        )
    optional = port.get("optional", False)
    if not isinstance(optional, bool) or optional:
        raise ValueError("present Hygon NN tensor ports must be non-optional")
    uid = port.get("uid")
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError(f"node {direction} UID is invalid")
    try:
        tensor = tensor_registry[uid]
    except KeyError as error:
        raise ValueError(
            f"node references unknown tensor UID {uid}"
        ) from error
    return uid, _validate_tensor(tensor, f"tensor {uid}")


def _resolved_ports(
    operation: str,
    schema: OperationSchema,
    parameters: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs = schema.input_ports
    outputs = schema.output_ports
    if schema.conditional_bias and _require_flag(parameters, "has_bias"):
        inputs += ("bias",)
    if schema.conditional_dbias and _require_flag(parameters, "has_dbias"):
        outputs += ("dbias",)
    if operation in ("sdpa_fp8", "sdpa_fp8_backward"):
        if "has_bias" in parameters and _require_flag(parameters, "has_bias"):
            raise ValueError(f"{operation} does not support bias")
        if "has_dbias" in parameters and _require_flag(
            parameters, "has_dbias"
        ):
            raise ValueError(f"{operation} does not support bias gradients")
    return inputs, outputs


def parse_node(
    node_value: object,
    position: int,
    node_count: int,
    tensor_registry: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Parse and semantically validate one schema-v3 DNN node."""

    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or position < 0
        or isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count <= 0
        or position >= node_count
    ):
        raise ValueError("graph node position/count is invalid")
    node = _require_object(node_value, f"graph.nodes[{position}]")
    node_id = node.get("id")
    if (
        isinstance(node_id, bool)
        or not isinstance(node_id, int)
        or node_id < 0
        or node_id >= node_count
    ):
        raise ValueError("graph node ID is invalid")
    operation = node.get("type")
    if not isinstance(operation, str) or operation not in OPERATION_SCHEMAS:
        raise ValueError(
            f"Hygon NN compiler does not support operation {operation!r}"
        )
    if node.get("compute_data_type") != "float32":
        raise ValueError(f"{operation} requires float32 compute_data_type")
    parameters = dict(
        _require_object(
            node.get("attributes"), f"graph.nodes[{position}].attributes"
        )
    )
    schema = OPERATION_SCHEMAS[operation]
    input_ports, output_ports = _resolved_ports(operation, schema, parameters)
    inputs = _require_sequence(node.get("inputs"), "node.inputs")
    outputs = _require_sequence(node.get("outputs"), "node.outputs")
    if len(inputs) != len(input_ports) or len(outputs) != len(output_ports):
        raise ValueError(f"{operation} node port count is invalid")

    input_uids: list[int] = []
    output_uids: list[int] = []
    tensors: list[Mapping[str, Any]] = []
    port_tensors: dict[str, Mapping[str, Any]] = {}
    port_indices: dict[str, int] = {}
    for index, name in enumerate(input_ports):
        uid, tensor = _parse_port(
            inputs[index], name, f"input[{index}]", tensor_registry
        )
        input_uids.append(uid)
        port_indices[name] = len(tensors)
        port_tensors[name] = tensor
        tensors.append(tensor)
    for index, name in enumerate(output_ports):
        uid, tensor = _parse_port(
            outputs[index], name, f"output[{index}]", tensor_registry
        )
        output_uids.append(uid)
        port_indices[name] = len(tensors)
        port_tensors[name] = tensor
        tensors.append(tensor)
    if len(set(output_uids)) != len(output_uids):
        raise ValueError(f"{operation} output tensors must be distinct")
    if set(input_uids).intersection(output_uids):
        raise ValueError(f"{operation} does not support in-place outputs")

    parsed = {
        "id": node_id,
        "operation": operation,
        "schema": schema,
        "compute_data_type": "float32",
        "parameters": parameters,
        "tensors": tensors,
        "tensor_roles": [*input_ports, *output_ports],
        "port_tensors": port_tensors,
        "port_indices": port_indices,
        "input_uids": input_uids,
        "output_uids": output_uids,
    }
    if operation in CONVOLUTION_OPERATIONS:
        derived = _validate_convolution(parsed)
    elif operation in NORMALIZATION_OPERATIONS:
        derived = _validate_normalization(parsed)
    else:
        derived = _validate_attention(parsed)
    parsed["derived"] = derived
    return parsed


def _same_dtype(
    tensors: Sequence[Mapping[str, Any]], allowed: frozenset[str], name: str
) -> str:
    data_types = {str(tensor["data_type"]) for tensor in tensors}
    if len(data_types) != 1:
        raise ValueError(f"{name} tensor data types must match")
    data_type = next(iter(data_types))
    if data_type not in allowed:
        raise ValueError(f"{name} data type {data_type!r} is unsupported")
    return data_type


def _metadata_array(
    parameters: Mapping[str, Any], name: str, expected: Sequence[int]
) -> list[int]:
    result = _require_integer_list(
        parameters, name, len(expected), minimum=1, maximum=MAX_I64
    )
    if result != list(expected):
        raise ValueError(
            f"parameters.{name} is inconsistent with tensor metadata"
        )
    return result


def _convolution_output_dimension(
    input_size: int,
    filter_size: int,
    pre_padding: int,
    post_padding: int,
    stride: int,
    dilation: int,
) -> int:
    effective = (filter_size - 1) * dilation + 1
    padded = input_size + pre_padding + post_padding
    if padded < effective:
        raise ValueError("convolution filter is larger than padded input")
    return (padded - effective) // stride + 1


def _validate_convolution(node: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(node["operation"])
    p = _require_object(node["parameters"], "node.parameters")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    spatial_rank = _require_integer(p, "spatial_rank", minimum=1, maximum=3)
    rank = spatial_rank + 2
    pre = _require_integer_list(p, "pre_padding", spatial_rank, minimum=0)
    post = _require_integer_list(p, "post_padding", spatial_rank, minimum=0)
    stride = _require_integer_list(p, "stride", spatial_rank, minimum=1)
    dilation = _require_integer_list(p, "dilation", spatial_rank, minimum=1)
    groups = _require_integer(p, "groups", minimum=1, maximum=MAX_I32)

    if operation in ("conv2d_fprop", "convolution_fprop"):
        image, weight, result = (
            ports["input"],
            ports["filter"],
            ports["output"],
        )
        if "convolution_mode" in p:
            mode = _require_integer(
                p, "convolution_mode", minimum=0, maximum=1
            )
            if mode != 0:
                raise ValueError(
                    "convolution FProp supports CROSS_CORRELATION only"
                )
        flip_filter = False
    elif operation == "convolution_dgrad":
        result, weight, image = ports["dy"], ports["w"], ports["dx"]
        flip_filter = bool(
            _require_integer(p, "convolution_mode", minimum=0, maximum=1)
        )
    else:
        result, image, weight = ports["dy"], ports["x"], ports["dw"]
        flip_filter = bool(
            _require_integer(p, "convolution_mode", minimum=0, maximum=1)
        )

    tensors = (image, weight, result)
    data_type = _same_dtype(tensors, FLOAT_TYPES, "convolution")
    if any(len(tensor["dimensions"]) != rank for tensor in tensors):
        raise ValueError(
            "convolution tensor ranks must equal spatial_rank + 2"
        )
    image_dims = list(image["dimensions"])
    weight_dims = list(weight["dimensions"])
    result_dims = list(result["dimensions"])
    batch, in_channels = image_dims[:2]
    out_channels = weight_dims[0]
    if in_channels % groups or out_channels % groups:
        raise ValueError("convolution channels must be divisible by groups")
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    if weight_dims[1] != in_per_group:
        raise ValueError("convolution filter channels do not match image")
    expected_result = [batch, out_channels]
    for axis in range(spatial_rank):
        expected_result.append(
            _convolution_output_dimension(
                image_dims[axis + 2],
                weight_dims[axis + 2],
                pre[axis],
                post[axis],
                stride[axis],
                dilation[axis],
            )
        )
    if result_dims != expected_result:
        raise ValueError("convolution output/loss shape is incorrect")
    n_outputs = _require_integer(p, "n_outputs", maximum=MAX_I64)
    expected_n_outputs = _checked_product(
        list(
            node["port_tensors"][
                (
                    "dx"
                    if operation == "convolution_dgrad"
                    else "dw" if operation == "convolution_wgrad" else "output"
                )
            ]["dimensions"]
        ),
        "convolution output elements",
    )
    if n_outputs != expected_n_outputs:
        raise ValueError(
            "parameters.n_outputs is inconsistent with the output tensor"
        )
    return {
        "data_type": data_type,
        "spatial_rank": spatial_rank,
        "pre_padding": pre,
        "post_padding": post,
        "stride": stride,
        "dilation": dilation,
        "groups": groups,
        "flip_filter": flip_filter,
        "image": image,
        "weight": weight,
        "result": result,
        "batch": batch,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "in_per_group": in_per_group,
        "out_per_group": out_per_group,
        "n_outputs": n_outputs,
    }


def _validate_normalization(node: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(node["operation"])
    p = _require_object(node["parameters"], "node.parameters")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    x = ports["x"]
    y = ports["y"]
    if x["data_type"] != y["data_type"] or x["data_type"] not in FLOAT_TYPES:
        raise ValueError("normalization X/Y must use one floating data type")
    if list(x["dimensions"]) != list(y["dimensions"]):
        raise ValueError("normalization Y shape must match X")

    if operation in ("layernorm", "rmsnorm"):
        scale, bias = ports["scale"], ports["bias"]
        _same_dtype((x, y, scale, bias), FLOAT_TYPES, "normalization")
        if not 1 <= len(x["dimensions"]) <= MAX_RANK:
            raise ValueError("normalization X rank must be in [1, 8]")
        if not _is_contiguous(x) or not _is_contiguous(y):
            raise ValueError("normalization X/Y must be contiguous")
        scale_dims = list(scale["dimensions"])
        x_dims = list(x["dimensions"])
        if not 1 <= len(scale_dims) <= len(x_dims):
            raise ValueError("normalization scale rank is invalid")
        leading = len(x_dims) - len(scale_dims)
        normalized_suffix = False
        normalized_elements = 1
        statistic_dims = list(x_dims)
        for axis, x_dim in enumerate(x_dims):
            scale_dim = 1 if axis < leading else scale_dims[axis - leading]
            if scale_dim != 1:
                if scale_dim != x_dim:
                    raise ValueError(
                        "normalization scale shape does not match X"
                    )
                normalized_suffix = True
            elif normalized_suffix and x_dim != 1:
                raise ValueError(
                    "normalization scale must describe a contiguous suffix"
                )
            if normalized_suffix:
                normalized_elements *= x_dim
                if normalized_elements > MAX_I64:
                    raise ValueError("normalization extent overflows int64")
                statistic_dims[axis] = 1
        if (
            not normalized_suffix
            or _checked_product(scale_dims, "normalization scale elements")
            != normalized_elements
            or _checked_product(
                list(bias["dimensions"]), "normalization bias elements"
            )
            != normalized_elements
        ):
            raise ValueError("normalization scale/bias size is invalid")
        if not _is_contiguous(scale) or not _is_contiguous(bias):
            raise ValueError("normalization scale/bias must be contiguous")
        statistic_names = (
            ("inv_variance",)
            if operation == "rmsnorm"
            else ("mean", "inv_variance")
        )
        for name in statistic_names:
            statistic = ports[name]
            if (
                statistic["data_type"] != "float32"
                or list(statistic["dimensions"]) != statistic_dims
                or not _is_contiguous(statistic)
            ):
                raise ValueError(
                    f"normalization statistic {name!r} metadata is invalid"
                )
        rows = _checked_product(x_dims, "normalization X elements")
        rows //= normalized_elements
        if _require_integer(p, "rows", maximum=MAX_I64) != rows:
            raise ValueError("parameters.rows is inconsistent with X")
        if (
            _require_integer(p, "normalized_elements", maximum=MAX_I64)
            != normalized_elements
        ):
            raise ValueError(
                "parameters.normalized_elements is inconsistent with scale"
            )
        epsilon = _require_number(p, "epsilon", positive=True)
        if (
            "forward_phase" in p
            and _require_integer(
                p, "forward_phase", minimum=0, maximum=MAX_I32
            )
            != 2
        ):
            raise ValueError("normalization supports TRAINING phase only")
        return {
            "data_type": str(x["data_type"]),
            "rows": rows,
            "normalized_elements": normalized_elements,
            "epsilon": epsilon,
            "contiguous": True,
        }

    if not 2 <= len(x["dimensions"]) <= MAX_RANK:
        raise ValueError("batchnorm X rank must be in [2, 8]")
    dims = list(x["dimensions"])
    batch, channels = dims[:2]
    spatial = _checked_product(dims[2:] or [1], "batchnorm spatial extent")
    n_elements = _checked_product(dims, "batchnorm elements")
    expected = {
        "n_elements": n_elements,
        "channels": channels,
        "spatial": spatial,
        "rank": len(dims),
    }
    if operation == "batchnorm":
        expected["batch"] = batch
    for name, value in expected.items():
        if _require_integer(p, name, maximum=MAX_I64) != value:
            raise ValueError(f"parameters.{name} is inconsistent with X")
    _metadata_array(p, "dimensions", dims)
    _metadata_array(p, "x_strides", list(x["strides"]))
    _metadata_array(p, "y_strides", list(y["strides"]))

    if operation == "batchnorm":
        scale, bias = ports["scale"], ports["bias"]
        _same_dtype((x, y, scale, bias), FLOAT_TYPES, "batchnorm")
        for name in ("scale", "bias"):
            tensor = ports[name]
            if _checked_product(
                list(tensor["dimensions"]), f"batchnorm {name} elements"
            ) != channels or not _is_contiguous(tensor):
                raise ValueError(
                    f"batchnorm {name} must be contiguous with C elements"
                )
        for name in (
            "previous_running_mean",
            "previous_running_variance",
            "mean",
            "inv_variance",
            "next_running_mean",
            "next_running_variance",
        ):
            tensor = ports[name]
            if (
                tensor["data_type"] != "float32"
                or _checked_product(
                    list(tensor["dimensions"]),
                    f"batchnorm {name} elements",
                )
                != channels
                or not _is_contiguous(tensor)
            ):
                raise ValueError(
                    f"batchnorm statistic {name!r} must be contiguous "
                    "float32[C]"
                )
        epsilon = _require_number(p, "epsilon", positive=True)
        momentum = _require_number(p, "momentum")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("batchnorm momentum must be in [0, 1]")
        return {
            "data_type": str(x["data_type"]),
            "batch": batch,
            "channels": channels,
            "spatial": spatial,
            "n_elements": n_elements,
            "epsilon": epsilon,
            "momentum": momentum,
            "contiguous": _is_contiguous(x) and _is_contiguous(y),
        }

    for name in ("mean", "inv_variance", "scale", "bias"):
        tensor = ports[name]
        if tensor["data_type"] not in FLOAT_TYPES:
            raise ValueError(
                f"batchnorm inference {name} must be floating point"
            )
        if _checked_product(
            list(tensor["dimensions"]),
            f"batchnorm inference {name} elements",
        ) != channels or not _is_contiguous(tensor):
            raise ValueError(
                f"batchnorm inference {name} must be contiguous with "
                "C elements"
            )
    return {
        "data_type": str(x["data_type"]),
        "batch": batch,
        "channels": channels,
        "spatial": spatial,
        "n_elements": n_elements,
        "epsilon": 0.0,
        "contiguous": _is_contiguous(x) and _is_contiguous(y),
    }


def _validate_qkv(
    q: Mapping[str, Any],
    k: Mapping[str, Any],
    v: Mapping[str, Any],
    *,
    fp8: bool,
) -> dict[str, int | str]:
    allowed = FP8_TYPES if fp8 else FLOAT_TYPES
    data_type = _same_dtype((q, k, v), allowed, "SDPA Q/K/V")
    if any(len(tensor["dimensions"]) != 4 for tensor in (q, k, v)):
        raise ValueError("SDPA Q/K/V must be rank-4 BHSD tensors")
    qd, kd, vd = (
        list(q["dimensions"]),
        list(k["dimensions"]),
        list(v["dimensions"]),
    )
    if qd[0] != kd[0] or qd[0] != vd[0]:
        raise ValueError("SDPA Q/K/V batch dimensions must match")
    if qd[3] != kd[3]:
        raise ValueError("SDPA Q/K head dimensions must match")
    if kd[2] != vd[2]:
        raise ValueError("SDPA K/V sequence dimensions must match")
    if qd[1] % kd[1] or qd[1] % vd[1]:
        raise ValueError(
            "SDPA query heads must be divisible by key/value heads"
        )
    if qd[3] > 256 or vd[3] > 256:
        raise ValueError(
            "SDPA head dimensions greater than 256 are unsupported"
        )
    return {
        "data_type": data_type,
        "batch": qd[0],
        "heads": qd[1],
        "key_heads": kd[1],
        "value_heads": vd[1],
        "sequence_q": qd[2],
        "sequence_kv": kd[2],
        "head_dimension": qd[3],
        "value_dimension": vd[3],
        "q_per_k": qd[1] // kd[1],
        "q_per_v": qd[1] // vd[1],
    }


def _validate_attention_tensor(
    tensor: Mapping[str, Any],
    expected_shape: Sequence[int],
    expected_type: str,
    name: str,
) -> None:
    if (
        tensor["data_type"] != expected_type
        or list(tensor["dimensions"]) != list(expected_shape)
        or len(tensor["dimensions"]) != 4
    ):
        raise ValueError(f"{name} tensor metadata is incorrect")


def _validate_bias(
    tensor: Mapping[str, Any],
    shape: Mapping[str, int | str],
    data_type: str,
    name: str,
) -> None:
    dims = list(tensor["dimensions"])
    if (
        tensor["data_type"] != data_type
        or len(dims) != 4
        or dims[0] not in (1, shape["batch"])
        or dims[1] not in (1, shape["heads"])
        or dims[2] != shape["sequence_q"]
        or dims[3] != shape["sequence_kv"]
    ):
        raise ValueError(
            f"{name} must broadcast over B/H and match Q/KV sequences"
        )


def _validate_fp32_scalar(tensor: Mapping[str, Any], name: str) -> None:
    if (
        tensor["data_type"] != "float32"
        or _checked_product(list(tensor["dimensions"]), name) != 1
    ):
        raise ValueError(f"{name} must be a one-element float32 tensor")


def _validate_attention(node: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(node["operation"])
    p = _require_object(node["parameters"], "node.parameters")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    fp8 = operation in ("sdpa_fp8", "sdpa_fp8_backward")
    shape = _validate_qkv(ports["q"], ports["k"], ports["v"], fp8=fp8)
    if operation == "sdpa_backward" and (
        shape["key_heads"] != shape["value_heads"]
    ):
        raise ValueError("SDPA backward requires matching K/V head counts")
    data_type = str(shape["data_type"])
    b = int(shape["batch"])
    h = int(shape["heads"])
    sq = int(shape["sequence_q"])
    d = int(shape["head_dimension"])
    dv = int(shape["value_dimension"])
    output_shape = [b, h, sq, dv]
    stats_shape = [b, h, sq, 1]

    _validate_attention_tensor(ports["o"], output_shape, data_type, "SDPA O")
    if operation in ("sdpa_backward", "sdpa_fp8_backward"):
        _validate_attention_tensor(
            ports["do"], output_shape, data_type, "SDPA dO"
        )
        for primal, gradient in (("q", "dq"), ("k", "dk"), ("v", "dv")):
            _validate_attention_tensor(
                ports[gradient],
                list(ports[primal]["dimensions"]),
                data_type,
                f"SDPA {gradient}",
            )
    stats = ports["stats"]
    if (
        stats["data_type"] != "float32"
        or list(stats["dimensions"]) != stats_shape
    ):
        raise ValueError("SDPA stats must be float32 [B,H,SQ,1]")

    has_bias = _require_flag(p, "has_bias")
    has_dbias = _require_flag(p, "has_dbias")
    if has_bias:
        _validate_bias(ports["bias"], shape, data_type, "SDPA bias")
    if has_dbias:
        _validate_bias(ports["dbias"], shape, data_type, "SDPA dbias")
    if operation != "sdpa_backward" and has_dbias:
        raise ValueError(f"{operation} cannot produce dbias")

    for name in (
        "batch",
        "heads",
        "key_heads",
        "value_heads",
        "sequence_q",
        "sequence_kv",
        "head_dimension",
        "value_dimension",
        "q_per_k",
        "q_per_v",
    ):
        if _require_integer(p, name, maximum=MAX_I64) != int(shape[name]):
            raise ValueError(
                f"parameters.{name} is inconsistent with Q/K/V metadata"
            )
    min_diag = _require_integer(
        p, "min_diag", minimum=-MAX_I32, maximum=MAX_I32
    )
    max_diag = _require_integer(
        p, "max_diag", minimum=-MAX_I32, maximum=MAX_I32
    )
    if min_diag > max_diag:
        raise ValueError("SDPA diagonal interval is empty")
    banded = _require_flag(p, "banded")
    if banded != (
        min_diag != -UNBOUNDED_DIAGONAL or max_diag != UNBOUNDED_DIAGONAL
    ):
        raise ValueError("parameters.banded disagrees with diagonal bounds")
    causal_top_left = _require_flag(p, "causal_top_left")
    reverse_causal = _require_flag(p, "reverse_causal")
    generate_stats = _require_flag(p, "generate_stats")
    if (
        operation in ("sdpa_backward", "sdpa_fp8_backward")
        and not generate_stats
    ):
        raise ValueError("SDPA backward requires generated forward stats")
    attn_scale = _require_number(p, "attn_scale")

    if fp8:
        scalar_names = (
            (
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_s",
                "scale_s",
                "scale_o",
            )
            if operation == "sdpa_fp8"
            else (
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_o",
                "descale_do",
                "descale_s",
                "descale_dp",
                "scale_s",
                "scale_dq",
                "scale_dk",
                "scale_dv",
                "scale_dp",
            )
        )
        for name in scalar_names:
            _validate_fp32_scalar(ports[name], f"FP8 SDPA {name}")
        amax_names = (
            ("amax_s", "amax_o")
            if operation == "sdpa_fp8"
            else ("amax_dq", "amax_dk", "amax_dv", "amax_dp")
        )
        for name in amax_names:
            _validate_fp32_scalar(ports[name], f"FP8 SDPA {name}")
    if operation == "sdpa_fp8_backward" and (
        shape["key_heads"] != shape["value_heads"] or d != dv or d > 128
    ):
        raise ValueError(
            "FP8 SDPA backward requires matching K/V heads and D == V <= 128"
        )
    return {
        **shape,
        "has_bias": has_bias,
        "has_dbias": has_dbias,
        "min_diag": min_diag,
        "max_diag": max_diag,
        "banded": banded,
        "causal_top_left": causal_top_left,
        "reverse_causal": reverse_causal,
        "generate_stats": generate_stats,
        "attn_scale": attn_scale,
    }


PointerArgument = tuple[str, str, str, str | int]
ScalarArgument = tuple[str, str]


def _make_stage(
    *,
    operation: str,
    stage_name: str,
    function_name: str,
    pointer_arguments: Sequence[PointerArgument],
    constants: Mapping[str, int | float | bool],
    tuning: TuningMetadata,
    tuning_key_value: int,
    grid_spec: GridSpec,
    scalar_arguments: Sequence[ScalarArgument] = (),
    dependencies: tuple[str, ...] = (),
    num_warps: int = 4,
    num_stages: int = 1,
    precision_mode: str = STRICT_DOT_INPUT_PRECISION,
) -> KernelStagePlan:
    if function_name not in REGISTRY_FUNCTIONS[operation]:
        raise ValueError(
            f"{function_name!r} is not a registry function for {operation}"
        )
    specialized = dict(constants)
    runtime_signature: dict[str, str] = {}
    runtime_values: dict[str, int | float] = {}
    layout: list[tuple[str, str | int | None]] = []
    for argument_name, token, kind, payload in pointer_arguments:
        if argument_name in runtime_signature or argument_name in specialized:
            raise ValueError(f"duplicate kernel argument {argument_name!r}")
        if not token.startswith("*"):
            raise ValueError(
                f"pointer argument {argument_name!r} is not a pointer"
            )
        if kind not in ("tensor", "workspace_tensor"):
            raise ValueError(f"invalid pointer ABI kind {kind!r}")
        runtime_signature[argument_name] = token
        layout.append((kind, payload))
    for argument_name, token in scalar_arguments:
        if argument_name in runtime_signature:
            raise ValueError(f"duplicate kernel argument {argument_name!r}")
        if argument_name not in specialized:
            raise ValueError(
                f"runtime scalar {argument_name!r} is missing from constants"
            )
        value = specialized.pop(argument_name)
        if token == "i32":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < MIN_I32
                or value > MAX_I32
            ):
                raise ValueError(
                    f"runtime scalar {argument_name!r}={value!r} cannot be "
                    "represented by the Hygon scalar_i32 artifact ABI"
                )
            runtime_value: int | float = int(value)
            layout_kind = "scalar_i32"
        elif token == "fp32":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"runtime scalar {argument_name!r} must be numeric"
                )
            runtime_value = float(value)
            if (
                not math.isfinite(runtime_value)
                or abs(runtime_value) > MAX_FLOAT32
            ):
                raise ValueError(
                    f"runtime scalar {argument_name!r}={value!r} cannot be "
                    "represented by the Hygon scalar_f32 artifact ABI"
                )
            layout_kind = "scalar_f32"
        else:
            raise ValueError(
                f"unsupported runtime scalar token {token!r} for "
                f"{argument_name!r}"
            )
        runtime_signature[argument_name] = token
        runtime_values[argument_name] = runtime_value
        layout.append((layout_kind, argument_name))
    stage_tuning = TuningMetadata(
        source=tuning.source,
        table=tuning.table,
        key=tuning.key,
        strategy=tuning.strategy,
        warmup=tuning.warmup,
        repetitions=tuning.repetitions,
        meta_keys=tuple(
            name for name in tuning.meta_keys if name in specialized
        ),
    )
    _validate_launch(num_warps, num_stages)
    default_grid = grid_spec.evaluate(specialized)
    stage = KernelStagePlan(
        operation=operation,
        stage_name=stage_name,
        function_name=function_name,
        runtime_signature=runtime_signature,
        runtime_values=runtime_values,
        constants=specialized,
        default_grid=default_grid,
        argument_layout=tuple(layout),
        tuning=stage_tuning,
        tuning_key_value=tuning_key_value,
        default_num_warps=num_warps,
        default_num_stages=num_stages,
        grid_spec=grid_spec,
        dependencies=dependencies,
        precision_mode=precision_mode,
    )
    stage.validate_launch()
    return stage


def _tensor_pointer(
    node: Mapping[str, Any], argument_name: str, port_name: str
) -> PointerArgument:
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    indices = _require_object(node["port_indices"], "node.port_indices")
    tensor = ports[port_name]
    return (
        argument_name,
        POINTER_TYPES[str(tensor["data_type"])],
        "tensor",
        int(indices[port_name]),
    )


def _workspace_pointer(
    argument_name: str, workspace_name: str, data_type: str = "float32"
) -> PointerArgument:
    return (
        argument_name,
        POINTER_TYPES[data_type],
        "workspace_tensor",
        workspace_name,
    )


def _padded_nd(
    tensor: Mapping[str, Any], spatial_rank: int
) -> tuple[list[int], list[int]]:
    dimensions = list(tensor["dimensions"])
    strides = list(tensor["strides"])
    leading = 3 - spatial_rank
    return (
        dimensions[:2] + [1] * leading + dimensions[2:],
        strides[:2] + [0] * leading + strides[2:],
    )


def _padded_spatial(values: Sequence[int], fill: int) -> list[int]:
    return [fill] * (3 - len(values)) + list(values)


def _convolution_stage(node: Mapping[str, Any]) -> KernelStagePlan:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    spatial_rank = int(d["spatial_rank"])
    image = d["image"]
    weight = d["weight"]
    result = d["result"]
    image_dims, image_strides = _padded_nd(image, spatial_rank)
    weight_dims, weight_strides = _padded_nd(weight, spatial_rank)
    result_dims, result_strides = _padded_nd(result, spatial_rank)
    _, _, xd, xh, xw = image_dims
    _, _, kd, kh, kw = weight_dims
    _, _, od, oh, ow = result_dims
    stride_d, stride_h, stride_w = _padded_spatial(d["stride"], 1)
    pad_front, pad_top, pad_left = _padded_spatial(d["pre_padding"], 0)
    dil_d, dil_h, dil_w = _padded_spatial(d["dilation"], 1)
    common: dict[str, int | float | bool] = {
        "XD": xd,
        "XH": xh,
        "XW": xw,
        "OD": od,
        "OH": oh,
        "OW": ow,
        "KD": kd,
        "KH": kh,
        "KW": kw,
        "CIN_PER_GROUP": int(d["in_per_group"]),
        "COUT_PER_GROUP": int(d["out_per_group"]),
        "STRIDE_D": stride_d,
        "STRIDE_H": stride_h,
        "STRIDE_W": stride_w,
        "PAD_FRONT": pad_front,
        "PAD_TOP": pad_top,
        "PAD_LEFT": pad_left,
        "DIL_D": dil_d,
        "DIL_H": dil_h,
        "DIL_W": dil_w,
        "INPUT_PRECISION": int(ALLOW_TF32),
    }
    for prefix, strides, axes in (
        ("X", image_strides, ("N", "C", "D", "H", "W")),
        ("W", weight_strides, ("K", "C", "D", "H", "W")),
        ("Y", result_strides, ("N", "C", "D", "H", "W")),
    ):
        for axis, value in zip(axes, strides, strict=True):
            common[f"{prefix}_STRIDE_{axis}"] = value

    if operation in ("conv2d_fprop", "convolution_fprop"):
        pointers = (
            _tensor_pointer(node, "x_ptr", "input"),
            _tensor_pointer(node, "w_ptr", "filter"),
            # Graph FProp has no bias port. HAS_BIAS=False makes this a safe
            # placeholder, while retaining the registry kernel's exact ABI.
            _tensor_pointer(node, "bias_ptr", "input"),
            _tensor_pointer(node, "y_ptr", "output"),
        )
        if spatial_rank == 1:
            constants = {
                "M": int(d["batch"]) * ow,
                "XL": xw,
                "OL": ow,
                "DTYPE_ID": DTYPE_IDS[str(d["data_type"])],
                "x_stride_n": image["strides"][0],
                "x_stride_c": image["strides"][1],
                "x_stride_l": image["strides"][2],
                "w_stride_o": weight["strides"][0],
                "w_stride_i": weight["strides"][1],
                "w_stride_k": weight["strides"][2],
                "bias_stride": 1,
                "y_stride_n": result["strides"][0],
                "y_stride_c": result["strides"][1],
                "y_stride_l": result["strides"][2],
                "CIN_PER_GROUP": int(d["in_per_group"]),
                "COUT_PER_GROUP": int(d["out_per_group"]),
                "KW": kw,
                "STRIDE_W": stride_w,
                "PAD_LEFT": pad_left,
                "DIL_W": dil_w,
                "HAS_BIAS": False,
                "BLOCK_M": 32,
                "BLOCK_OC": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "INPUT_PRECISION": int(ALLOW_TF32),
            }
            function = "conv1d_gemm_kernel"
            grid = GridSpec(
                "conv1d",
                (
                    int(d["batch"]) * ow,
                    int(d["out_per_group"]),
                    int(d["groups"]),
                ),
            )
        elif spatial_rank == 2:
            constants = {
                "XH": xh,
                "XW": xw,
                "OH": oh,
                "OW": ow,
                "C_IN": int(d["in_channels"]),
                "C_OUT": int(d["out_channels"]),
                "CIN_PER_GROUP": int(d["in_per_group"]),
                "COUT_PER_GROUP": int(d["out_per_group"]),
                "GROUPS": int(d["groups"]),
                "STRIDE_H": stride_h,
                "STRIDE_W": stride_w,
                "PAD_TOP": pad_top,
                "PAD_LEFT": pad_left,
                "DIL_H": dil_h,
                "DIL_W": dil_w,
                "KH": kh,
                "KW": kw,
                "HAS_BIAS": False,
                "BLOCK_OC": 32,
                "BLOCK_HW": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "DTYPE_ID": DTYPE_IDS[str(d["data_type"])],
                "INPUT_PRECISION": int(ALLOW_TF32),
                "X_STRIDE_N": image_strides[0],
                "X_STRIDE_C": image_strides[1],
                "X_STRIDE_H": image_strides[3],
                "X_STRIDE_W": image_strides[4],
                "W_STRIDE_K": weight_strides[0],
                "W_STRIDE_C": weight_strides[1],
                "W_STRIDE_R": weight_strides[3],
                "W_STRIDE_S": weight_strides[4],
                "Y_STRIDE_N": result_strides[0],
                "Y_STRIDE_C": result_strides[1],
                "Y_STRIDE_H": result_strides[3],
                "Y_STRIDE_W": result_strides[4],
            }
            function = "conv2d_spatial_nchw_kernel"
            grid = GridSpec(
                "conv2d",
                (
                    oh * ow,
                    int(d["out_per_group"]),
                    int(d["batch"]) * int(d["groups"]),
                ),
            )
        else:
            constants = dict(common)
            constants.update(
                {
                    "M": int(d["batch"]) * od * oh * ow,
                    "C_IN": int(d["in_channels"]),
                    "C_OUT": int(d["out_channels"]),
                    "HAS_BIAS": False,
                    "BLOCK_OC": 32,
                    "BLOCK_M": 32,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                }
            )
            function = "conv3d_spatial_ncdhw_m_kernel"
            grid = GridSpec(
                "conv3d",
                (
                    int(d["batch"]) * od * oh * ow,
                    int(d["out_per_group"]),
                    int(d["groups"]),
                ),
            )
    else:
        loss_port = "dy"
        first_port = "w" if operation == "convolution_dgrad" else "x"
        output_port = "dx" if operation == "convolution_dgrad" else "dw"
        pointers = (
            _tensor_pointer(node, "dy_ptr", loss_port),
            _tensor_pointer(
                node,
                "w_ptr" if operation == "convolution_dgrad" else "x_ptr",
                first_port,
            ),
            _tensor_pointer(
                node,
                "dx_ptr" if operation == "convolution_dgrad" else "dw_ptr",
                output_port,
            ),
        )
        # Backward common kernels consistently call the loss tensor DY and
        # image/output tensor X, independent of which one is the graph output.
        constants = dict(common)
        for axis in ("N", "C", "D", "H", "W"):
            constants[f"DY_STRIDE_{axis}"] = constants.pop(f"Y_STRIDE_{axis}")
        constants["FLIP_FILTER"] = bool(d["flip_filter"])
        if operation == "convolution_dgrad":
            constants.update(
                {
                    "M": int(d["batch"]) * xd * xh * xw,
                    "BLOCK_M": 32,
                    "BLOCK_CI": 32,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                }
            )
            function = "conv_dgrad_nd_kernel"
            grid = GridSpec(
                "conv_dgrad",
                (
                    int(d["batch"]) * xd * xh * xw,
                    int(d["in_per_group"]),
                    int(d["groups"]),
                ),
            )
        else:
            constants.update(
                {
                    "M": int(d["batch"]) * od * oh * ow,
                    "BLOCK_OC": 32,
                    "BLOCK_CI": 32,
                    "BLOCK_M": 32,
                }
            )
            function = "conv_wgrad_nd_kernel"
            grid = GridSpec(
                "conv_wgrad",
                (
                    int(d["out_per_group"]),
                    int(d["in_per_group"]),
                    kd * kh * kw,
                    int(d["groups"]),
                ),
            )
    return _make_stage(
        operation=operation,
        stage_name=operation,
        function_name=function,
        pointer_arguments=pointers,
        constants=constants,
        tuning=CONV_TUNING_BY_OPERATION[operation],
        tuning_key_value=int(d["n_outputs"]),
        grid_spec=grid,
        num_stages=2,
    )


def _contiguous_strides(dimensions: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(dimensions):
        result.append(stride)
        stride *= int(dimension)
    return tuple(reversed(result))


def _aligned_size(size: int) -> int:
    return _ceil_div(size, WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT


def _private_workspace(
    name: str,
    data_type: str,
    dimensions: Sequence[int],
    *,
    offset: int = 0,
) -> WorkspaceTensor:
    typed_dimensions = tuple(int(value) for value in dimensions)
    elements = _checked_product(
        list(typed_dimensions), f"{name} workspace elements"
    )
    size = elements * ELEMENT_SIZES[data_type]
    if size <= 0 or offset < 0 or offset + size > MAX_I64:
        raise ValueError(f"{name} workspace size is invalid")
    return WorkspaceTensor(
        name=name,
        data_type=data_type,
        dimensions=typed_dimensions,
        strides=_contiguous_strides(typed_dimensions),
        offset=offset,
        size=size,
    )


def _private_fprop_plan(node: Mapping[str, Any]) -> NodePlan | None:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    if int(d["spatial_rank"]) != 2:
        return None
    image = d["image"]
    weight = d["weight"]
    result = d["result"]
    if not all(_is_contiguous(tensor) for tensor in (image, weight, result)):
        return None
    n, c_in, xh, xw = (int(value) for value in image["dimensions"])
    c_out, _, kh, kw = (int(value) for value in weight["dimensions"])
    _, _, oh, ow = (int(value) for value in result["dimensions"])
    groups = int(d["groups"])
    cin_per_group = int(d["in_per_group"])
    cout_per_group = int(d["out_per_group"])
    stride = list(d["stride"])
    pre = list(d["pre_padding"])
    post = list(d["post_padding"])
    dilation = list(d["dilation"])
    flip_filter = bool(d["flip_filter"])

    standard_3x3_fp32 = (
        str(d["data_type"]) == "float32"
        and n == 8
        and c_in == 32
        and c_out == 64
        and xh == 32
        and xw == 32
        and oh == 32
        and ow == 32
        and kh == 3
        and kw == 3
        and stride == [1, 1]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and groups == 1
        and not flip_filter
    )
    if standard_3x3_fp32:
        stage = _make_stage(
            operation=operation,
            stage_name="fprop_standard_3x3",
            function_name="hygon_conv2d_fprop_standard_3x3_nchw_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "x_ptr", "input"),
                _tensor_pointer(node, "w_ptr", "filter"),
                _tensor_pointer(node, "y_ptr", "output"),
            ),
            constants={
                "BLOCK_OC_S": 64,
                "BLOCK_K_S": 32,
            },
            tuning=CONV_FPROP_STANDARD_3X3_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_fprop_standard_3x3", (oh, c_out, n)
            ),
            num_warps=4,
            num_stages=1,
        )
        return NodePlan(operation, (stage,))

    yolo_x_p5_fp32_gemm = (
        str(d["data_type"]) == "float32"
        and n == 1
        and c_in == 768
        and c_out == 768
        and xh == 40
        and xw == 40
        and oh == 20
        and ow == 20
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and groups == 1
        and not flip_filter
    )

    unit_1x1 = (
        kh == 1
        and kw == 1
        and stride == [1, 1]
        and pre == [0, 0]
        and post == [0, 0]
        and dilation == [1, 1]
    )
    if unit_1x1:
        block_oc, block_m, block_ci = 32, 32, 32
        stage = _make_stage(
            operation=operation,
            stage_name="fprop_1x1_nchw",
            function_name="hygon_conv2d_fprop_1x1_nchw_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "x_ptr", "input"),
                _tensor_pointer(node, "w_ptr", "filter"),
                _tensor_pointer(node, "y_ptr", "output"),
            ),
            constants={
                "HW": oh * ow,
                "C_IN": c_in,
                "C_OUT": c_out,
                "CIN_PER_GROUP": cin_per_group,
                "COUT_PER_GROUP": cout_per_group,
                "GROUPS": groups,
                "BLOCK_OC": block_oc,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
            },
            tuning=CONV_FPROP_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_output", (oh * ow, cout_per_group, n * groups)
            ),
            num_warps=4,
            num_stages=2,
        )
        return NodePlan(operation, (stage,))

    low_ci_stride2_3x3 = (
        groups == 1
        and cin_per_group <= 3
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and n * c_in * xh * xw <= MAX_I32
        and c_out * cin_per_group * kh * kw <= MAX_I32
        and n * c_out * oh * ow <= MAX_I32
    )
    if low_ci_stride2_3x3:
        block_oc, block_m, block_k = 64, 32, 32
        stage = _make_stage(
            operation=operation,
            stage_name="fprop_stride2_low_ci",
            function_name="hygon_conv2d_fprop_stride2_low_ci_nchw_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "x_ptr", "input"),
                _tensor_pointer(node, "w_ptr", "filter"),
                _tensor_pointer(node, "y_ptr", "output"),
            ),
            constants={
                "XH": xh,
                "XW": xw,
                "OH": oh,
                "OW": ow,
                "C_IN": c_in,
                "C_OUT": c_out,
                "BLOCK_OC": block_oc,
                "BLOCK_M": block_m,
                "BLOCK_K": block_k,
            },
            tuning=CONV_FPROP_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec("conv_private_fprop_rows", (oh, ow, c_out, n)),
            num_warps=4,
            num_stages=1,
        )
        return NodePlan(operation, (stage,))

    # The packed matrix is group-local. Keep grouped convolution on the
    # explicit-stride generic fallback until a grouped workspace ABI exists.
    if groups != 1:
        return None
    output_area = oh * ow
    reduction_extent = cin_per_group * kh * kw

    columns = _private_workspace(
        "fprop_columns",
        str(d["data_type"]),
        (n, reduction_extent, output_area),
    )
    # Bound provider-local memory so unusual graphs cannot turn this
    # optimization into an unbounded allocation policy.
    if columns.size > 512 * 1024 * 1024:
        return None
    col_stride_n, col_stride_k, col_stride_m = columns.strides
    pack_block_m, pack_block_k = 64, 16
    pack = _make_stage(
        operation=operation,
        stage_name="fprop_im2col",
        function_name="hygon_conv2d_im2col_nchw_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "x_ptr", "input"),
            _workspace_pointer(
                "col_ptr", "fprop_columns", str(d["data_type"])
            ),
        ),
        constants={
            "M": output_area,
            "XH": xh,
            "XW": xw,
            "OH": oh,
            "OW": ow,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_TOP": pre[0],
            "PAD_LEFT": pre[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "X_STRIDE_N": int(image["strides"][0]),
            "X_STRIDE_C": int(image["strides"][1]),
            "X_STRIDE_H": int(image["strides"][2]),
            "X_STRIDE_W": int(image["strides"][3]),
            "COL_STRIDE_N": col_stride_n,
            "COL_STRIDE_K": col_stride_k,
            "COL_STRIDE_M": col_stride_m,
            "BLOCK_M": pack_block_m,
            "BLOCK_K": pack_block_k,
        },
        tuning=CONV_FPROP_TUNING,
        tuning_key_value=output_area,
        grid_spec=GridSpec(
            "conv_private_im2col", (output_area, reduction_extent, n)
        ),
        num_warps=4,
    )
    block_oc, block_m, block_k = 32, 32, 32
    gemm = _make_stage(
        operation=operation,
        stage_name="fprop_gemm",
        function_name=(
            "hygon_conv2d_fprop_yolo_x_p5_gemm_kernel"
            if yolo_x_p5_fp32_gemm
            else "hygon_conv2d_fprop_im2col_kernel"
        ),
        pointer_arguments=(
            _tensor_pointer(node, "w_ptr", "filter"),
            _workspace_pointer(
                "col_ptr", "fprop_columns", str(d["data_type"])
            ),
            _tensor_pointer(node, "y_ptr", "output"),
        ),
        constants={
            "M": output_area,
            "COUT_PER_GROUP": cout_per_group,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "W_STRIDE_K": int(weight["strides"][0]),
            "W_STRIDE_C": int(weight["strides"][1]),
            "W_STRIDE_H": int(weight["strides"][2]),
            "W_STRIDE_W": int(weight["strides"][3]),
            "Y_STRIDE_N": int(result["strides"][0]),
            "Y_STRIDE_C": int(result["strides"][1]),
            "Y_STRIDE_H": int(result["strides"][2]),
            "Y_STRIDE_W": int(result["strides"][3]),
            "OW": ow,
            "COL_STRIDE_N": col_stride_n,
            "COL_STRIDE_K": col_stride_k,
            "COL_STRIDE_M": col_stride_m,
            **(
                {
                    "BLOCK_OC_X": block_oc,
                    "BLOCK_M_X": block_m,
                    "BLOCK_K_X": block_k,
                    "GROUP_M_X": 8,
                    "YOLO_X_P5": 1,
                }
                if yolo_x_p5_fp32_gemm
                else {
                    "BLOCK_OC": block_oc,
                    "BLOCK_M": block_m,
                    "BLOCK_K": block_k,
                    "GROUP_M": 4,
                }
            ),
        },
        tuning=(
            CONV_FPROP_YOLO_X_P5_GEMM_TUNING
            if yolo_x_p5_fp32_gemm
            else CONV_FPROP_TUNING
        ),
        tuning_key_value=int(d["n_outputs"]),
        grid_spec=GridSpec(
            (
                "conv_private_fprop_yolo_x"
                if yolo_x_p5_fp32_gemm
                else "conv_private_output"
            ),
            (output_area, cout_per_group, n),
        ),
        dependencies=("fprop_im2col",),
        num_warps=4,
        num_stages=2,
    )
    return NodePlan(
        operation,
        (pack, gemm),
        (columns,),
        _aligned_size(columns.size),
    )


def _private_dgrad_plan(node: Mapping[str, Any]) -> NodePlan | None:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    if int(d["spatial_rank"]) != 2:
        return None
    image = d["image"]
    weight = d["weight"]
    loss = d["result"]
    if not all(_is_contiguous(tensor) for tensor in (image, weight, loss)):
        return None
    n, c_in, xh, xw = (int(value) for value in image["dimensions"])
    c_out, _, kh, kw = (int(value) for value in weight["dimensions"])
    _, _, oh, ow = (int(value) for value in loss["dimensions"])
    groups = int(d["groups"])
    cin_per_group = int(d["in_per_group"])
    cout_per_group = int(d["out_per_group"])
    stride = list(d["stride"])
    pre = list(d["pre_padding"])
    post = list(d["post_padding"])
    dilation = list(d["dilation"])
    flip_filter = bool(d["flip_filter"])

    unit_1x1 = (
        kh == 1
        and kw == 1
        and stride == [1, 1]
        and pre == [0, 0]
        and post == [0, 0]
        and dilation == [1, 1]
    )
    if unit_1x1:
        block_m, block_ci, block_co = 32, 32, 32
        stage = _make_stage(
            operation=operation,
            stage_name="dgrad_1x1_nchw",
            function_name="hygon_conv_dgrad2d_1x1_nchw_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "w_ptr", "w"),
                _tensor_pointer(node, "dx_ptr", "dx"),
            ),
            constants={
                "HW": xh * xw,
                "C_IN": c_in,
                "C_OUT": c_out,
                "CIN_PER_GROUP": cin_per_group,
                "COUT_PER_GROUP": cout_per_group,
                "GROUPS": groups,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_CO": block_co,
            },
            tuning=CONV_DGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_dgrad", (xh * xw, cin_per_group, n * groups)
            ),
            num_warps=4,
            num_stages=2,
        )
        return NodePlan(operation, (stage,))

    stride2_3x3 = (
        kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and xh >= 2
        and xw >= 2
    )
    if stride2_3x3:
        if groups == 1:
            output_area = oh * ow
            packed_extent = c_in * kh * kw
            block_pointer_gemm = c_in > 4 and not flip_filter
            column_dimensions = (
                (n, packed_extent, output_area)
                if block_pointer_gemm
                else (n, output_area, packed_extent)
            )
            columns = _private_workspace(
                "dgrad_s2_contribution_columns",
                "float32",
                column_dimensions,
            )
            if columns.size > 512 * 1024 * 1024:
                return None
            if block_pointer_gemm:
                col_stride_n, col_stride_k, col_stride_m = columns.strides
            else:
                col_stride_n, col_stride_m, col_stride_k = columns.strides
            gemm_function = (
                "hygon_conv_dgrad2d_stride2_block_pointer_gemm_kernel"
                if block_pointer_gemm
                else "hygon_conv_dgrad2d_stride2_contribution_gemm_kernel"
            )
            gemm_constants = {
                "M": output_area,
                "C_IN": c_in,
                "C_OUT": c_out,
                "KH": kh,
                "KW": kw,
                "FLIP_FILTER": flip_filter,
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "OW": ow,
                "COL_STRIDE_N": col_stride_n,
                "COL_STRIDE_M": col_stride_m,
                "COL_STRIDE_K": col_stride_k,
                "BLOCK_M": 32,
                "BLOCK_CI": 32,
                "BLOCK_CO": 32,
            }
            if block_pointer_gemm:
                gemm_constants["GROUP_M"] = 4
            gemm = _make_stage(
                operation=operation,
                stage_name="dgrad_s2_contribution_gemm",
                function_name=gemm_function,
                pointer_arguments=(
                    _tensor_pointer(node, "dy_ptr", "dy"),
                    _tensor_pointer(node, "w_ptr", "w"),
                    _workspace_pointer(
                        "col_ptr", "dgrad_s2_contribution_columns", "float32"
                    ),
                ),
                constants=gemm_constants,
                tuning=CONV_DGRAD_TUNING,
                tuning_key_value=int(d["n_outputs"]),
                grid_spec=GridSpec(
                    "conv_private_dgrad_columns",
                    (output_area, packed_extent, n),
                ),
                num_warps=4,
                num_stages=2,
            )
            n_elements = n * c_in * xh * xw
            col2im = _make_stage(
                operation=operation,
                stage_name="dgrad_s2_contribution_col2im",
                function_name=(
                    "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel"
                ),
                pointer_arguments=(
                    _workspace_pointer(
                        "col_ptr", "dgrad_s2_contribution_columns", "float32"
                    ),
                    _tensor_pointer(node, "dx_ptr", "dx"),
                ),
                constants={
                    "N_ELEMENTS": n_elements,
                    "XH": xh,
                    "XW": xw,
                    "OH": oh,
                    "OW": ow,
                    "C_IN": c_in,
                    "KH": kh,
                    "KW": kw,
                    "PAD_TOP": pre[0],
                    "PAD_LEFT": pre[1],
                    "DIL_H": dilation[0],
                    "DIL_W": dilation[1],
                    "COL_STRIDE_N": col_stride_n,
                    "COL_STRIDE_M": col_stride_m,
                    "COL_STRIDE_K": col_stride_k,
                    "X_STRIDE_N": int(image["strides"][0]),
                    "X_STRIDE_C": int(image["strides"][1]),
                    "X_STRIDE_H": int(image["strides"][2]),
                    "X_STRIDE_W": int(image["strides"][3]),
                    "BLOCK_SIZE": 256,
                },
                tuning=CONV_DGRAD_TUNING,
                tuning_key_value=int(d["n_outputs"]),
                grid_spec=GridSpec("linear", (n_elements,)),
                dependencies=("dgrad_s2_contribution_gemm",),
                num_warps=4,
            )
            return NodePlan(
                operation,
                (gemm, col2im),
                (columns,),
                _aligned_size(columns.size),
            )

        block_m, block_ci, block_co = 32, 32, 32
        stages: list[KernelStagePlan] = []
        previous: tuple[str, ...] = ()
        for parity_h in range(2):
            for parity_w in range(2):
                count_h = (xh - parity_h + 1) // 2
                count_w = (xw - parity_w + 1) // 2
                parity_m = count_h * count_w
                stage_name = f"dgrad_s2_p{parity_h}{parity_w}"
                stage = _make_stage(
                    operation=operation,
                    stage_name=stage_name,
                    function_name=("hygon_conv_dgrad2d_stride2_parity_kernel"),
                    pointer_arguments=(
                        _tensor_pointer(node, "dy_ptr", "dy"),
                        _tensor_pointer(node, "w_ptr", "w"),
                        _tensor_pointer(node, "dx_ptr", "dx"),
                    ),
                    constants={
                        "PARITY_M": parity_m,
                        "PARITY_W_COUNT": count_w,
                        "PARITY_H": parity_h,
                        "PARITY_W": parity_w,
                        "XH": xh,
                        "XW": xw,
                        "OH": oh,
                        "OW": ow,
                        "CIN_PER_GROUP": cin_per_group,
                        "COUT_PER_GROUP": cout_per_group,
                        "GROUPS": groups,
                        "KH": kh,
                        "KW": kw,
                        "PAD_TOP": pre[0],
                        "PAD_LEFT": pre[1],
                        "DIL_H": dilation[0],
                        "DIL_W": dilation[1],
                        "FLIP_FILTER": flip_filter,
                        "DY_STRIDE_N": int(loss["strides"][0]),
                        "DY_STRIDE_C": int(loss["strides"][1]),
                        "DY_STRIDE_H": int(loss["strides"][2]),
                        "DY_STRIDE_W": int(loss["strides"][3]),
                        "W_STRIDE_K": int(weight["strides"][0]),
                        "W_STRIDE_C": int(weight["strides"][1]),
                        "W_STRIDE_H": int(weight["strides"][2]),
                        "W_STRIDE_W": int(weight["strides"][3]),
                        "X_STRIDE_N": int(image["strides"][0]),
                        "X_STRIDE_C": int(image["strides"][1]),
                        "X_STRIDE_H": int(image["strides"][2]),
                        "X_STRIDE_W": int(image["strides"][3]),
                        "BLOCK_M": block_m,
                        "BLOCK_CI": block_ci,
                        "BLOCK_CO": block_co,
                    },
                    tuning=CONV_DGRAD_TUNING,
                    tuning_key_value=int(d["n_outputs"]),
                    grid_spec=GridSpec(
                        "conv_private_dgrad",
                        (parity_m, cin_per_group, n * groups),
                    ),
                    dependencies=previous,
                    num_warps=4,
                    num_stages=2,
                )
                stages.append(stage)
                previous = (stage_name,)
        return NodePlan(operation, tuple(stages))

    exact_stride1_3x3 = (
        str(d["data_type"]) in ("float32", "bfloat16")
        and kh == 3
        and kw == 3
        and stride == [1, 1]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and groups == 1
        and not flip_filter
        and n == 8
        and c_in == 32
        and c_out == 64
        and xh == 32
        and xw == 32
        and oh == 32
        and ow == 32
    )
    if exact_stride1_3x3:
        output_area = oh * ow
        packed_extent = c_in * kh * kw
        columns = _private_workspace(
            "dgrad_s1_exact_columns",
            "float32",
            (n, packed_extent, output_area),
        )
        workspace_size = _aligned_size(columns.size)
        if (
            columns.offset != 0
            or columns.alignment != WORKSPACE_ALIGNMENT
            or columns.size != DGRAD_EXACT_3X3_S1_WORKSPACE_CAP
            or workspace_size != DGRAD_EXACT_3X3_S1_WORKSPACE_CAP
        ):
            return None
        col_stride_n, col_stride_k, col_stride_m = columns.strides

        # BLOCK_CI/BLOCK_CO are intentionally fixed: the exact GEMM has no
        # tail masks. EXACT_3X3_S1 keeps generic DGrad rows out of this route.
        gemm = _make_stage(
            operation=operation,
            stage_name="dgrad_s1_exact_gemm",
            function_name="hygon_conv_dgrad2d_exact_3x3_s1_gemm_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "w_ptr", "w"),
                _workspace_pointer(
                    "col_ptr", "dgrad_s1_exact_columns", "float32"
                ),
            ),
            constants={
                "M": output_area,
                "C_IN": c_in,
                "C_OUT": c_out,
                "KH": kh,
                "KW": kw,
                "FLIP_FILTER": flip_filter,
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "OW": ow,
                "COL_STRIDE_N": col_stride_n,
                "COL_STRIDE_M": col_stride_m,
                "COL_STRIDE_K": col_stride_k,
                "EXACT_3X3_S1": 1,
                "BLOCK_M": 64,
                "BLOCK_CI": 32,
                "BLOCK_CO": 32,
                "GROUP_M": 16,
            },
            tuning=CONV_DGRAD_EXACT_GEMM_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_dgrad_columns",
                (output_area, packed_extent, n),
            ),
            num_warps=4,
            num_stages=1,
        )
        col2im = _make_stage(
            operation=operation,
            stage_name="dgrad_s1_exact_col2im",
            function_name="hygon_conv_dgrad2d_exact_3x3_s1_col2im_kernel",
            pointer_arguments=(
                _workspace_pointer(
                    "col_ptr", "dgrad_s1_exact_columns", "float32"
                ),
                _tensor_pointer(node, "dx_ptr", "dx"),
            ),
            constants={
                "M": output_area,
                "XH": xh,
                "XW": xw,
                "OH": oh,
                "OW": ow,
                "C_IN": c_in,
                "KH": kh,
                "KW": kw,
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "COL_STRIDE_N": col_stride_n,
                "COL_STRIDE_M": col_stride_m,
                "COL_STRIDE_K": col_stride_k,
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "EXACT_3X3_S1": 1,
                "BLOCK_H": 4,
            },
            tuning=CONV_DGRAD_EXACT_COL2IM_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec("conv_private_dgrad_fixed_2d", (xh, n * c_in)),
            dependencies=("dgrad_s1_exact_gemm",),
            num_warps=2,
            num_stages=1,
        )
        return NodePlan(
            operation,
            (gemm, col2im),
            (columns,),
            workspace_size,
        )

    if stride == [1, 1] and kh * kw <= 25:
        block_m, block_ci, block_co = 32, 32, 32
        stage = _make_stage(
            operation=operation,
            stage_name="dgrad_stride1",
            function_name="hygon_conv_dgrad2d_stride1_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "w_ptr", "w"),
                _tensor_pointer(node, "dx_ptr", "dx"),
            ),
            constants={
                "M": n * xh * xw,
                "XH": xh,
                "XW": xw,
                "OH": oh,
                "OW": ow,
                "CIN_PER_GROUP": cin_per_group,
                "COUT_PER_GROUP": cout_per_group,
                "KH": kh,
                "KW": kw,
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "FLIP_FILTER": flip_filter,
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_CO": block_co,
            },
            tuning=CONV_DGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_dgrad", (n * xh * xw, cin_per_group, groups)
            ),
            num_warps=4,
            num_stages=2,
        )
        return NodePlan(operation, (stage,))
    return None


def _wgrad_reduce_stage(
    node: Mapping[str, Any],
    *,
    workspace_name: str,
    num_splits: int,
    cout: int,
    cin_per_group: int,
    kh: int,
    kw: int,
    partial_strides: Sequence[int],
    dependency: str,
) -> KernelStagePlan:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    weight = d["weight"]
    cik = cin_per_group * kh * kw
    total = cout * cik
    block_size = 256
    return _make_stage(
        operation=operation,
        stage_name="wgrad_reduce",
        function_name="hygon_conv_wgrad2d_reduce_kernel",
        pointer_arguments=(
            _workspace_pointer("partial_ptr", workspace_name, "float32"),
            _tensor_pointer(node, "dw_ptr", "dw"),
        ),
        constants={
            "TOTAL": total,
            "CIK": cik,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "FLIP_FILTER": bool(d["flip_filter"]),
            "NUM_SPLITS": num_splits,
            "PARTIAL_STRIDE_SPLIT": int(partial_strides[0]),
            "PARTIAL_STRIDE_OC": int(partial_strides[1]),
            "PARTIAL_STRIDE_K": int(partial_strides[2]),
            "W_STRIDE_K": int(weight["strides"][0]),
            "W_STRIDE_C": int(weight["strides"][1]),
            "W_STRIDE_H": int(weight["strides"][2]),
            "W_STRIDE_W": int(weight["strides"][3]),
            "BLOCK_SIZE": block_size,
        },
        tuning=CONV_WGRAD_TUNING,
        tuning_key_value=total,
        grid_spec=GridSpec("linear", (total,)),
        dependencies=(dependency,),
        num_warps=4,
    )


def _private_wgrad_plan(node: Mapping[str, Any]) -> NodePlan | None:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    if int(d["spatial_rank"]) != 2:
        return None
    image = d["image"]
    weight = d["weight"]
    loss = d["result"]
    if not all(_is_contiguous(tensor) for tensor in (image, weight, loss)):
        return None
    n, c_in, xh, xw = (int(value) for value in image["dimensions"])
    c_out, _, kh, kw = (int(value) for value in weight["dimensions"])
    _, _, oh, ow = (int(value) for value in loss["dimensions"])
    groups = int(d["groups"])
    cin_per_group = int(d["in_per_group"])
    cout_per_group = int(d["out_per_group"])
    stride = list(d["stride"])
    pre = list(d["pre_padding"])
    post = list(d["post_padding"])
    dilation = list(d["dilation"])
    output_area = oh * ow
    total_rows = n * output_area

    unit_1x1 = (
        kh == 1
        and kw == 1
        and stride == [1, 1]
        and pre == [0, 0]
        and post == [0, 0]
        and dilation == [1, 1]
    )
    if unit_1x1 and total_rows >= 2048:
        num_splits = 8 if total_rows >= 4096 else 4
        partial = _private_workspace(
            "wgrad_partial",
            "float32",
            (num_splits, c_out, cin_per_group),
        )
        if partial.size > 512 * 1024 * 1024:
            return None
        block_oc, block_ci, block_m = 16, 16, 64
        split = _make_stage(
            operation=operation,
            stage_name="wgrad_1x1_split",
            function_name="hygon_conv_wgrad2d_1x1_split_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
            ),
            constants={
                "TOTAL_ROWS": total_rows,
                "ROWS_PER_SPLIT": _ceil_div(total_rows, num_splits),
                "HW": output_area,
                "C_IN": c_in,
                "C_OUT": c_out,
                "CIN_PER_GROUP": cin_per_group,
                "COUT_PER_GROUP": cout_per_group,
                "GROUPS": groups,
                "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
                "PARTIAL_STRIDE_OC": int(partial.strides[1]),
                "PARTIAL_STRIDE_K": int(partial.strides[2]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI": block_ci,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad_1x1",
                (cout_per_group, cin_per_group, num_splits * groups),
            ),
            num_warps=4,
            num_stages=2,
        )
        reduce = _wgrad_reduce_stage(
            node,
            workspace_name="wgrad_partial",
            num_splits=num_splits,
            cout=c_out,
            cin_per_group=cin_per_group,
            kh=kh,
            kw=kw,
            partial_strides=partial.strides,
            dependency="wgrad_1x1_split",
        )
        return NodePlan(
            operation,
            (split, reduce),
            (partial,),
            _aligned_size(partial.size),
        )

    stem_split_3x3 = (
        n == 1
        and groups == 1
        and cin_per_group <= 3
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and dilation == [1, 1]
        and total_rows >= 65536
        and oh % 64 == 0
    )
    if stem_split_3x3:
        num_splits = 64
        reduction_extent = cin_per_group * kh * kw
        partial = _private_workspace(
            "wgrad_partial",
            "float32",
            (num_splits, c_out, reduction_extent),
        )
        if partial.size > 512 * 1024 * 1024:
            return None
        block_oc, block_ci_k, block_m = 64, 32, 64
        split = _make_stage(
            operation=operation,
            stage_name="wgrad_stem_split",
            function_name="hygon_conv_wgrad2d_stem_split_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
            ),
            constants={
                "OUTPUT_ROWS_PER_SPLIT": oh // num_splits,
                "OH": oh,
                "OW": ow,
                "XH": xh,
                "XW": xw,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "STRIDE_H": stride[0],
                "STRIDE_W": stride[1],
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
                "PARTIAL_STRIDE_OC": int(partial.strides[1]),
                "PARTIAL_STRIDE_K": int(partial.strides[2]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad",
                (cout_per_group, reduction_extent, num_splits),
            ),
            num_warps=4,
            num_stages=2,
        )
        reduce = _wgrad_reduce_stage(
            node,
            workspace_name="wgrad_partial",
            num_splits=num_splits,
            cout=c_out,
            cin_per_group=cin_per_group,
            kh=kh,
            kw=kw,
            partial_strides=partial.strides,
            dependency="wgrad_stem_split",
        )
        return NodePlan(
            operation,
            (split, reduce),
            (partial,),
            _aligned_size(partial.size),
        )

    multirow_split_3x3 = (
        n > 1
        and groups == 1
        and cin_per_group > 3
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and not bool(d["flip_filter"])
        and ow <= 32
    )
    if multirow_split_3x3:
        num_splits = n
        reduction_extent = cin_per_group * kh * kw
        partial = _private_workspace(
            "wgrad_partial",
            "float32",
            (num_splits, c_out, reduction_extent),
        )
        if partial.size > 512 * 1024 * 1024:
            return None
        block_oc, block_ci_k, block_m = 128, 64, 64
        split = _make_stage(
            operation=operation,
            stage_name="wgrad_multirow_split",
            function_name="hygon_conv_wgrad2d_multirow_split_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
            ),
            constants={
                "OH": oh,
                "OW": ow,
                "XH": xh,
                "XW": xw,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "STRIDE_H": stride[0],
                "STRIDE_W": stride[1],
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
                "PARTIAL_STRIDE_OC": int(partial.strides[1]),
                "PARTIAL_STRIDE_K": int(partial.strides[2]),
                "ROW_PITCH": 32,
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad",
                (cout_per_group, reduction_extent, num_splits),
            ),
            num_warps=8,
            num_stages=2,
        )
        reduce = _wgrad_reduce_stage(
            node,
            workspace_name="wgrad_partial",
            num_splits=num_splits,
            cout=c_out,
            cin_per_group=cin_per_group,
            kh=kh,
            kw=kw,
            partial_strides=partial.strides,
            dependency="wgrad_multirow_split",
        )
        return NodePlan(
            operation,
            (split, reduce),
            (partial,),
            _aligned_size(partial.size),
        )

    p5_rowmajor_3x3 = (
        n == 1
        and groups == 1
        and cin_per_group > 3
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and not bool(d["flip_filter"])
        and xh == 40
        and xw == 40
        and oh == 20
        and ow == 20
    )
    if p5_rowmajor_3x3:
        reduction_extent = cin_per_group * kh * kw
        # gfx936 measurements show that block pointers win for the large
        # ML/X matrices. Keep N/S and tail shapes on the existing row-major
        # kernel so this optimization cannot regress their established path.
        large_p5_block_ptr = cin_per_group >= 512 and cout_per_group >= 512
        columns = _private_workspace(
            "wgrad_rowmajor_columns",
            str(d["data_type"]),
            (total_rows, reduction_extent),
        )
        if columns.size > 512 * 1024 * 1024:
            return None
        col_stride_r, col_stride_k = columns.strides
        pack_block_m, pack_block_k = 16, 64
        pack = _make_stage(
            operation=operation,
            stage_name="wgrad_im2row",
            function_name="hygon_conv_wgrad2d_im2row_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer(
                    "col_ptr",
                    "wgrad_rowmajor_columns",
                    str(d["data_type"]),
                ),
            ),
            constants={
                "M": output_area,
                "XH": xh,
                "XW": xw,
                "OW": ow,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "STRIDE_H": stride[0],
                "STRIDE_W": stride[1],
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "COL_STRIDE_R": col_stride_r,
                "COL_STRIDE_K": col_stride_k,
                "BLOCK_M": pack_block_m,
                "BLOCK_K": pack_block_k,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=total_rows * reduction_extent,
            grid_spec=GridSpec(
                "conv_private_im2col", (output_area, reduction_extent, n)
            ),
            num_warps=4,
        )
        block_oc, block_ci_k, block_m = 64, 64, 64
        if large_p5_block_ptr:
            gemm_function = "hygon_conv_wgrad2d_p5_block_ptr_kernel"
            gemm_constants = {
                "M": output_area,
                "COUT_PER_GROUP": cout_per_group,
                "REDUCTION_EXTENT": reduction_extent,
                "DY_STRIDE_C": int(loss["strides"][1]),
                "COL_STRIDE_R": col_stride_r,
                "COL_STRIDE_K": col_stride_k,
                "W_STRIDE_K": int(weight["strides"][0]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            }
        else:
            gemm_function = "hygon_conv_wgrad2d_rowmajor_kernel"
            gemm_constants = {
                "M": output_area,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "FLIP_FILTER": bool(d["flip_filter"]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "OW": ow,
                "COL_STRIDE_R": col_stride_r,
                "COL_STRIDE_K": col_stride_k,
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            }
        gemm = _make_stage(
            operation=operation,
            stage_name="wgrad_rowmajor",
            function_name=gemm_function,
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _workspace_pointer(
                    "col_ptr",
                    "wgrad_rowmajor_columns",
                    str(d["data_type"]),
                ),
                _tensor_pointer(node, "dw_ptr", "dw"),
            ),
            constants=gemm_constants,
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad",
                (cout_per_group, reduction_extent, 1),
            ),
            dependencies=("wgrad_im2row",),
            num_warps=8,
            num_stages=2,
        )
        return NodePlan(
            operation,
            (pack, gemm),
            (columns,),
            _aligned_size(columns.size),
        )

    direct_split_3x3 = (
        groups == 1
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and dilation == [1, 1]
        and total_rows >= 4096
    )
    if direct_split_3x3:
        num_splits = 64 if total_rows >= 65536 else 16
        reduction_extent = cin_per_group * kh * kw
        partial = _private_workspace(
            "wgrad_partial",
            "float32",
            (num_splits, c_out, reduction_extent),
        )
        if partial.size > 512 * 1024 * 1024:
            return None
        block_oc, block_ci_k, block_m = 16, 16, 64
        split = _make_stage(
            operation=operation,
            stage_name="wgrad_direct_split",
            function_name="hygon_conv_wgrad2d_direct_split_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
            ),
            constants={
                "TOTAL_ROWS": total_rows,
                "ROWS_PER_SPLIT": _ceil_div(total_rows, num_splits),
                "M": output_area,
                "XH": xh,
                "XW": xw,
                "OW": ow,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "STRIDE_H": stride[0],
                "STRIDE_W": stride[1],
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
                "PARTIAL_STRIDE_OC": int(partial.strides[1]),
                "PARTIAL_STRIDE_K": int(partial.strides[2]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad",
                (cout_per_group, reduction_extent, num_splits),
            ),
            num_warps=4,
            num_stages=2,
        )
        reduce = _wgrad_reduce_stage(
            node,
            workspace_name="wgrad_partial",
            num_splits=num_splits,
            cout=c_out,
            cin_per_group=cin_per_group,
            kh=kh,
            kw=kw,
            partial_strides=partial.strides,
            dependency="wgrad_direct_split",
        )
        return NodePlan(
            operation,
            (split, reduce),
            (partial,),
            _aligned_size(partial.size),
        )

    # The current packed workspace is deliberately group-one. All other
    # layouts and ranks remain covered by conv_wgrad_nd_kernel.
    if groups != 1 or unit_1x1:
        return None
    reduction_extent = cin_per_group * kh * kw
    columns = _private_workspace(
        "wgrad_columns",
        str(d["data_type"]),
        (n, reduction_extent, output_area),
    )
    if columns.size > 512 * 1024 * 1024:
        return None
    col_stride_n, col_stride_k, col_stride_m = columns.strides
    pack_block_m, pack_block_k = 64, 16
    pack = _make_stage(
        operation=operation,
        stage_name="wgrad_im2col",
        function_name="hygon_conv2d_im2col_nchw_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "x_ptr", "x"),
            _workspace_pointer(
                "col_ptr", "wgrad_columns", str(d["data_type"])
            ),
        ),
        constants={
            "M": output_area,
            "XH": xh,
            "XW": xw,
            "OH": oh,
            "OW": ow,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_TOP": pre[0],
            "PAD_LEFT": pre[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "X_STRIDE_N": int(image["strides"][0]),
            "X_STRIDE_C": int(image["strides"][1]),
            "X_STRIDE_H": int(image["strides"][2]),
            "X_STRIDE_W": int(image["strides"][3]),
            "COL_STRIDE_N": col_stride_n,
            "COL_STRIDE_K": col_stride_k,
            "COL_STRIDE_M": col_stride_m,
            "BLOCK_M": pack_block_m,
            "BLOCK_K": pack_block_k,
        },
        tuning=CONV_WGRAD_TUNING,
        tuning_key_value=output_area,
        grid_spec=GridSpec(
            "conv_private_im2col", (output_area, reduction_extent, n)
        ),
        num_warps=4,
    )
    parameter_elements = c_out * reduction_extent
    if total_rows >= 65536 and parameter_elements <= 131072:
        num_splits = 64
    elif total_rows >= 4096 and parameter_elements <= 131072:
        num_splits = 8
    else:
        num_splits = 1
    block_oc, block_ci_k, block_m = 16, 16, 64

    if num_splits == 1:
        gemm = _make_stage(
            operation=operation,
            stage_name="wgrad_gemm",
            function_name="hygon_conv_wgrad2d_im2col_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _workspace_pointer(
                    "col_ptr", "wgrad_columns", str(d["data_type"])
                ),
                _tensor_pointer(node, "dw_ptr", "dw"),
            ),
            constants={
                "N": n,
                "M": output_area,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "FLIP_FILTER": bool(d["flip_filter"]),
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "OW": ow,
                "COL_STRIDE_N": col_stride_n,
                "COL_STRIDE_K": col_stride_k,
                "COL_STRIDE_M": col_stride_m,
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad", (cout_per_group, reduction_extent, 1)
            ),
            dependencies=("wgrad_im2col",),
            num_warps=4,
            num_stages=2,
        )
        return NodePlan(
            operation,
            (pack, gemm),
            (columns,),
            _aligned_size(columns.size),
        )

    partial_offset = _aligned_size(columns.size)
    partial = _private_workspace(
        "wgrad_partial",
        "float32",
        (num_splits, c_out, reduction_extent),
        offset=partial_offset,
    )
    workspace_size = _aligned_size(partial.offset + partial.size)
    if workspace_size > 512 * 1024 * 1024:
        return None
    split = _make_stage(
        operation=operation,
        stage_name="wgrad_split",
        function_name="hygon_conv_wgrad2d_im2col_split_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "dy_ptr", "dy"),
            _workspace_pointer(
                "col_ptr", "wgrad_columns", str(d["data_type"])
            ),
            _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
        ),
        constants={
            "TOTAL_ROWS": total_rows,
            "ROWS_PER_SPLIT": _ceil_div(total_rows, num_splits),
            "M": output_area,
            "COUT_PER_GROUP": cout_per_group,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "DY_STRIDE_N": int(loss["strides"][0]),
            "DY_STRIDE_C": int(loss["strides"][1]),
            "DY_STRIDE_H": int(loss["strides"][2]),
            "DY_STRIDE_W": int(loss["strides"][3]),
            "OW": ow,
            "COL_STRIDE_N": col_stride_n,
            "COL_STRIDE_K": col_stride_k,
            "COL_STRIDE_M": col_stride_m,
            "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
            "PARTIAL_STRIDE_OC": int(partial.strides[1]),
            "PARTIAL_STRIDE_K": int(partial.strides[2]),
            "BLOCK_OC": block_oc,
            "BLOCK_CI_K": block_ci_k,
            "BLOCK_M": block_m,
        },
        tuning=CONV_WGRAD_TUNING,
        tuning_key_value=int(d["n_outputs"]),
        grid_spec=GridSpec(
            "conv_private_wgrad",
            (cout_per_group, reduction_extent, num_splits),
        ),
        dependencies=("wgrad_im2col",),
        num_warps=4,
        num_stages=2,
    )
    reduce = _wgrad_reduce_stage(
        node,
        workspace_name="wgrad_partial",
        num_splits=num_splits,
        cout=c_out,
        cin_per_group=cin_per_group,
        kh=kh,
        kw=kw,
        partial_strides=partial.strides,
        dependency="wgrad_split",
    )
    return NodePlan(
        operation,
        (pack, split, reduce),
        (columns, partial),
        workspace_size,
    )


def _convolution_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    if operation in ("conv2d_fprop", "convolution_fprop"):
        private = _private_fprop_plan(node)
    elif operation == "convolution_dgrad":
        private = _private_dgrad_plan(node)
    else:
        private = _private_wgrad_plan(node)
    if private is not None:
        private.validate_dependencies()
        return private
    return NodePlan(operation, (_convolution_stage(node),))


def _padded_rank_metadata(
    tensor: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    leading = MAX_RANK - len(tensor["dimensions"])
    return (
        [1] * leading + list(tensor["dimensions"]),
        [0] * leading + list(tensor["strides"]),
    )


def _normalization_stage(node: Mapping[str, Any]) -> KernelStagePlan:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    if operation == "layernorm":
        block = min(_next_power_of_two(int(d["normalized_elements"])), 65536)
        constants: dict[str, int | float | bool] = {
            "M": int(d["rows"]),
            "eps": float(d["epsilon"]),
            "N": int(d["normalized_elements"]),
            "BLOCK_SIZE": block,
            "ROWS_PER_PROGRAM": 1,
            "HAS_WEIGHT": True,
            "HAS_BIAS": True,
            "RETURN_STATS": True,
        }
        pointers = (
            _tensor_pointer(node, "x_ptr", "x"),
            _tensor_pointer(node, "y_ptr", "y"),
            _tensor_pointer(node, "mean_ptr", "mean"),
            _tensor_pointer(node, "inv_variance_ptr", "inv_variance"),
            _tensor_pointer(node, "weight_ptr", "scale"),
            _tensor_pointer(node, "bias_ptr", "bias"),
        )
        return _make_stage(
            operation=operation,
            stage_name=operation,
            function_name="layer_norm_kernel",
            pointer_arguments=pointers,
            constants=constants,
            scalar_arguments=(("M", "i32"),),
            tuning=LAYER_NORM_TUNING,
            tuning_key_value=int(d["normalized_elements"]),
            grid_spec=GridSpec("norm_rows", (int(d["rows"]),)),
        )
    if operation == "rmsnorm":
        block = min(_next_power_of_two(int(d["normalized_elements"])), 65536)
        constants = {
            "M": int(d["rows"]),
            "N": int(d["normalized_elements"]),
            "eps": float(d["epsilon"]),
            "BLOCK_SIZE": block,
            "ROWS_PER_PROGRAM": 1,
            "STATIC_ROWS": 0,
            "HAS_WEIGHT": True,
            "HAS_BIAS": True,
            "RETURN_STATS": True,
        }
        pointers = (
            _tensor_pointer(node, "x_ptr", "x"),
            _tensor_pointer(node, "y_ptr", "y"),
            _tensor_pointer(node, "weight_ptr", "scale"),
            _tensor_pointer(node, "bias_ptr", "bias"),
            _tensor_pointer(node, "inv_variance_ptr", "inv_variance"),
        )
        return _make_stage(
            operation=operation,
            stage_name=operation,
            function_name="rms_norm_kernel",
            pointer_arguments=pointers,
            constants=constants,
            scalar_arguments=(("M", "i32"),),
            tuning=RMS_NORM_TUNING,
            tuning_key_value=int(d["normalized_elements"]),
            grid_spec=GridSpec("norm_rows", (int(d["rows"]),)),
        )

    if operation == "batchnorm":
        batch_block = _next_power_of_two(int(d["batch"]))
        use_nchw = bool(d["contiguous"]) and batch_block <= 512
        block = max(256, batch_block) if use_nchw else 256
        constants = {
            "N": int(d["batch"]),
            "C": int(d["channels"]),
            "S": int(d["spatial"]),
            "eps": float(d["epsilon"]),
            "momentum": float(d["momentum"]),
            "BLOCK_SIZE": block,
            "IS_TRAINING": True,
            "HAS_WEIGHT": True,
            "HAS_BIAS": True,
            "HAS_RUNNING_STATS": True,
            "RETURN_STATS": True,
        }
        pointers = (
            _tensor_pointer(node, "x_ptr", "x"),
            _tensor_pointer(node, "y_ptr", "y"),
            _tensor_pointer(node, "mean_ptr", "previous_running_mean"),
            _tensor_pointer(node, "var_ptr", "previous_running_variance"),
            _tensor_pointer(node, "weight_ptr", "scale"),
            _tensor_pointer(node, "bias_ptr", "bias"),
            _tensor_pointer(node, "saved_mean_ptr", "mean"),
            _tensor_pointer(node, "saved_inv_var_ptr", "inv_variance"),
            _tensor_pointer(
                node, "next_running_mean_ptr", "next_running_mean"
            ),
            _tensor_pointer(
                node, "next_running_var_ptr", "next_running_variance"
            ),
        )
        function = (
            "batch_norm_nchw_kernel" if use_nchw else "batch_norm_kernel"
        )
        if not use_nchw:
            x = node["port_tensors"]["x"]
            y = node["port_tensors"]["y"]
            dimensions, x_strides = _padded_rank_metadata(x)
            _, y_strides = _padded_rank_metadata(y)
            constants["STRIDED"] = not bool(d["contiguous"])
            for axis in range(MAX_RANK):
                constants[f"DIM_{axis}"] = dimensions[axis]
                constants[f"INPUT_STRIDE_{axis}"] = x_strides[axis]
                constants[f"OUTPUT_STRIDE_{axis}"] = y_strides[axis]
        return _make_stage(
            operation=operation,
            stage_name=operation,
            function_name=function,
            pointer_arguments=pointers,
            constants=constants,
            scalar_arguments=(
                (("N", "i32"), ("C", "i32"), ("S", "i32"))
                if not use_nchw
                else ()
            ),
            tuning=BATCH_NORM_TUNING,
            tuning_key_value=int(d["channels"]),
            grid_spec=GridSpec("batchnorm_channels", (int(d["channels"]),)),
        )

    x = node["port_tensors"]["x"]
    y = node["port_tensors"]["y"]
    constants = {
        "total_elements": int(d["n_elements"]),
        "C": int(d["channels"]),
        "S": int(d["spatial"]),
        "eps": 0.0,
        "BLOCK_SIZE": 256,
        "HAS_WEIGHT": True,
        "HAS_BIAS": True,
        "STAT_IS_INV_VARIANCE": True,
    }
    pointers = (
        _tensor_pointer(node, "x_ptr", "x"),
        _tensor_pointer(node, "mean_ptr", "mean"),
        _tensor_pointer(node, "stat_ptr", "inv_variance"),
        _tensor_pointer(node, "weight_ptr", "scale"),
        _tensor_pointer(node, "bias_ptr", "bias"),
        _tensor_pointer(node, "y_ptr", "y"),
    )

    if bool(d["contiguous"]):
        function = "batch_norm_inference_nchw_kernel"
        constants.pop("total_elements")
        grid = GridSpec(
            "batchnorm_inference_nchw",
            (int(d["batch"]), int(d["channels"]), int(d["spatial"])),
        )
    else:
        function = "batch_norm_inference_kernel"
        dimensions, x_strides = _padded_rank_metadata(x)
        _, y_strides = _padded_rank_metadata(y)
        constants["STRIDED"] = True
        for axis in range(MAX_RANK):
            constants[f"DIM_{axis}"] = dimensions[axis]
            constants[f"INPUT_STRIDE_{axis}"] = x_strides[axis]
            constants[f"OUTPUT_STRIDE_{axis}"] = y_strides[axis]
        grid = GridSpec("linear", (int(d["n_elements"]),))
    return _make_stage(
        operation=operation,
        stage_name=operation,
        function_name=function,
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=(
            (
                ("total_elements", "i32"),
                ("C", "i32"),
                ("S", "i32"),
            )
            if not bool(d["contiguous"])
            else ()
        ),
        tuning=BATCH_NORM_INFERENCE_TUNING,
        tuning_key_value=int(d["n_elements"]),
        grid_spec=grid,
    )


LOG2_E = 1.4426950408889634

_ATTENTION_FORWARD_I32_SCALARS = (
    "HQ",
    "SQ",
    "SKV",
    "q_per_k",
    "q_per_v",
    "min_diag",
    "max_diag",
    "stride_qb",
    "stride_qh",
    "stride_qm",
    "stride_qd",
    "stride_kb",
    "stride_kh",
    "stride_kn",
    "stride_kd",
    "stride_vb",
    "stride_vh",
    "stride_vn",
    "stride_vd",
    "stride_bias_b",
    "stride_bias_h",
    "stride_bias_m",
    "stride_bias_n",
    "stride_ob",
    "stride_oh",
    "stride_om",
    "stride_od",
    "stride_sb",
    "stride_sh",
    "stride_sm",
)
_ATTENTION_BACKWARD_RUNTIME_SCALARS = (
    ("attn_scale", "fp32"),
    ("SQ", "i32"),
    ("SKV", "i32"),
    ("min_diag", "i32"),
    ("max_diag", "i32"),
)


def _attention_forward_scalar_arguments(
    *, fp8: bool
) -> tuple[ScalarArgument, ...]:
    scale_name = "attn_scale" if fp8 else "qk_scale"
    return ((scale_name, "fp32"),) + tuple(
        (name, "i32") for name in _ATTENTION_FORWARD_I32_SCALARS
    )


def _attention_bias_strides(
    node: Mapping[str, Any], d: Mapping[str, Any]
) -> tuple[int, int, int, int]:
    if not bool(d["has_bias"]):
        return (0, 0, 0, 0)
    bias = node["port_tensors"]["bias"]
    dims = list(bias["dimensions"])
    strides = list(bias["strides"])
    return (
        0 if dims[0] == 1 else strides[0],
        0 if dims[1] == 1 else strides[1],
        strides[2],
        strides[3],
    )


def _attention_forward_constants(
    node: Mapping[str, Any], *, fp8: bool
) -> dict[str, int | float | bool]:
    d = _require_object(node["derived"], "node.derived")
    q = node["port_tensors"]["q"]
    k = node["port_tensors"]["k"]
    v = node["port_tensors"]["v"]
    o = node["port_tensors"]["o"]
    stats = node["port_tensors"]["stats"]
    bias_strides = _attention_bias_strides(node, d)
    head_dimension = int(d["head_dimension"])
    value_dimension = int(d["value_dimension"])
    constants: dict[str, int | float | bool] = {
        ("attn_scale" if fp8 else "qk_scale"): (
            float(d["attn_scale"]) if fp8 else float(d["attn_scale"]) * LOG2_E
        ),
        "HQ": int(d["heads"]),
        "SQ": int(d["sequence_q"]),
        "SKV": int(d["sequence_kv"]),
        "q_per_k": int(d["q_per_k"]),
        "q_per_v": int(d["q_per_v"]),
        "min_diag": int(d["min_diag"]),
        "max_diag": int(d["max_diag"]),
        "stride_qb": q["strides"][0],
        "stride_qh": q["strides"][1],
        "stride_qm": q["strides"][2],
        "stride_qd": q["strides"][3],
        "stride_kb": k["strides"][0],
        "stride_kh": k["strides"][1],
        "stride_kn": k["strides"][2],
        "stride_kd": k["strides"][3],
        "stride_vb": v["strides"][0],
        "stride_vh": v["strides"][1],
        "stride_vn": v["strides"][2],
        "stride_vd": v["strides"][3],
        "stride_bias_b": bias_strides[0],
        "stride_bias_h": bias_strides[1],
        "stride_bias_m": bias_strides[2],
        "stride_bias_n": bias_strides[3],
        "stride_ob": o["strides"][0],
        "stride_oh": o["strides"][1],
        "stride_om": o["strides"][2],
        "stride_od": o["strides"][3],
        "stride_sb": stats["strides"][0],
        "stride_sh": stats["strides"][1],
        "stride_sm": stats["strides"][2],
        "HEAD_DIM": head_dimension,
        "V_DIM": value_dimension,
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_D": _attention_full_dimension_block(head_dimension),
        "BLOCK_DV": _attention_full_dimension_block(value_dimension),
        "HAS_BIAS": bool(d["has_bias"]),
        "BANDED": bool(d["banded"]),
        "GENERATE_STATS": bool(d["generate_stats"]),
        "REVERSE_CAUSAL": bool(d["reverse_causal"]),
    }
    if not fp8:
        constants["ELEM_SIZE"] = ELEMENT_SIZES[str(d["data_type"])]
    return constants


def _attention_forward_stage(
    node: Mapping[str, Any],
    *,
    dependencies: tuple[str, ...] = (),
) -> KernelStagePlan:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "stats_ptr", "stats"),
    )
    return _make_stage(
        operation=operation,
        stage_name="forward",
        function_name="_sdpa_fwd_kernel",
        pointer_arguments=pointers,
        constants=_attention_forward_constants(node, fp8=False),
        scalar_arguments=_attention_forward_scalar_arguments(fp8=False),
        tuning=SDPA_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_fwd",
            (
                int(d["sequence_q"]),
                int(d["batch"]) * int(d["heads"]),
            ),
        ),
        dependencies=dependencies,
        num_stages=2,
    )


def _fp8_forward_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = "sdpa_fp8"
    d = _require_object(node["derived"], "node.derived")
    zero = _make_stage(
        operation=operation,
        stage_name="zero_amax",
        function_name="_zero_sdpa_fp8_fwd_amax_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "amax_s_ptr", "amax_s"),
            _tensor_pointer(node, "amax_o_ptr", "amax_o"),
        ),
        constants={},
        tuning=INTERNAL_TUNING,
        tuning_key_value=1,
        grid_spec=GridSpec("fixed", (1, 1, 1)),
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _tensor_pointer(node, "amax_s_ptr", "amax_s"),
        _tensor_pointer(node, "amax_o_ptr", "amax_o"),
        _tensor_pointer(node, "descale_q_ptr", "descale_q"),
        _tensor_pointer(node, "descale_k_ptr", "descale_k"),
        _tensor_pointer(node, "descale_v_ptr", "descale_v"),
        _tensor_pointer(node, "descale_s_ptr", "descale_s"),
        _tensor_pointer(node, "scale_s_ptr", "scale_s"),
        _tensor_pointer(node, "scale_o_ptr", "scale_o"),
    )
    forward = _make_stage(
        operation=operation,
        stage_name="forward",
        function_name="_sdpa_fp8_fwd_kernel",
        pointer_arguments=pointers,
        constants=_attention_forward_constants(node, fp8=True),
        scalar_arguments=_attention_forward_scalar_arguments(fp8=True),
        tuning=SDPA_FP8_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_fp8_fwd",
            (
                int(d["sequence_q"]),
                int(d["batch"]) * int(d["heads"]),
            ),
        ),
        dependencies=("zero_amax",),
        num_stages=2,
    )
    plan = NodePlan(operation, (zero, forward))
    plan.validate_dependencies()
    return plan


def _attention_backward_base(
    node: Mapping[str, Any],
) -> dict[str, int | float | bool]:
    d = _require_object(node["derived"], "node.derived")
    q = node["port_tensors"]["q"]
    k = node["port_tensors"]["k"]
    v = node["port_tensors"]["v"]
    do = node["port_tensors"]["do"]
    stats = node["port_tensors"]["stats"]
    bias_strides = _attention_bias_strides(node, d)
    return {
        "SQ": int(d["sequence_q"]),
        "SKV": int(d["sequence_kv"]),
        "min_diag": int(d["min_diag"]),
        "max_diag": int(d["max_diag"]),
        "stride_qb": q["strides"][0],
        "stride_qh": q["strides"][1],
        "stride_qm": q["strides"][2],
        "stride_qd": q["strides"][3],
        "stride_kb": k["strides"][0],
        "stride_kh": k["strides"][1],
        "stride_kn": k["strides"][2],
        "stride_kd": k["strides"][3],
        "stride_vb": v["strides"][0],
        "stride_vh": v["strides"][1],
        "stride_vn": v["strides"][2],
        "stride_vd": v["strides"][3],
        "stride_bias_b": bias_strides[0],
        "stride_bias_h": bias_strides[1],
        "stride_bias_m": bias_strides[2],
        "stride_bias_n": bias_strides[3],
        "stride_dob": do["strides"][0],
        "stride_doh": do["strides"][1],
        "stride_dom": do["strides"][2],
        "stride_dod": do["strides"][3],
        "stride_sb": stats["strides"][0],
        "stride_sh": stats["strides"][1],
        "stride_sm": stats["strides"][2],
        "HEAD_DIM": int(d["head_dimension"]),
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_D_FULL": _attention_full_dimension_block(
            int(d["head_dimension"])
        ),
        # The common backward kernels iterate a full BLOCK_M query tile and
        # require masking for tail query blocks, even for non-banded attention.
        "FULL_ATTENTION": False,
        "HAS_BIAS": bool(d["has_bias"]),
        "BANDED": bool(d["banded"]),
        "CAUSAL_TOP_LEFT": bool(d["causal_top_left"]),
    }


def _attention_backward_dq_stage(
    node: Mapping[str, Any],
    dependencies: tuple[str, ...],
) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    o = node["port_tensors"]["o"]
    dq = node["port_tensors"]["dq"]
    delta_strides = (
        int(d["heads"]) * int(d["sequence_q"]),
        int(d["sequence_q"]),
        1,
    )
    has_dbias = bool(d["has_dbias"])
    dbias = node["port_tensors"].get("dbias")
    dbias_reduce = bool(
        has_dbias
        and (
            dbias["dimensions"][0] != d["batch"]
            or dbias["dimensions"][1] != d["heads"]
        )
    )
    dbias_strides = list(dbias["strides"]) if has_dbias else [0, 0, 0, 0]
    constants = _attention_backward_base(node)
    constants.update(
        {
            "attn_scale": float(d["attn_scale"]),
            "HQ": int(d["heads"]),
            "q_per_k": int(d["q_per_k"]),
            "q_per_v": int(d["q_per_v"]),
            "stride_ob": o["strides"][0],
            "stride_oh": o["strides"][1],
            "stride_om": o["strides"][2],
            "stride_od": o["strides"][3],
            "stride_delta_b": delta_strides[0],
            "stride_delta_h": delta_strides[1],
            "stride_delta_m": delta_strides[2],
            "stride_dqb": dq["strides"][0],
            "stride_dqh": dq["strides"][1],
            "stride_dqm": dq["strides"][2],
            "stride_dqd": dq["strides"][3],
            "stride_dbias_b": dbias_strides[0],
            "stride_dbias_h": dbias_strides[1],
            "stride_dbias_m": dbias_strides[2],
            "stride_dbias_n": dbias_strides[3],
            "V_DIM": int(d["value_dimension"]),
            "DBIAS_BATCHES": dbias["dimensions"][0] if has_dbias else 1,
            "DBIAS_HEADS": dbias["dimensions"][1] if has_dbias else 1,
            "BLOCK_D_OUT": _attention_output_dimension_block(
                int(d["head_dimension"])
            ),
            "BLOCK_DV": _attention_full_dimension_block(
                int(d["value_dimension"])
            ),
            "HAS_DBIAS": has_dbias,
            "DBIAS_REDUCE": dbias_reduce,
        }
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    dbias_port = "dbias" if has_dbias else "dq"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _workspace_pointer("delta_ptr", "delta"),
        _tensor_pointer(node, "dq_ptr", "dq"),
        _tensor_pointer(node, "dbias_ptr", dbias_port),
    )
    return _make_stage(
        operation="sdpa_backward",
        stage_name="dq_delta_dbias",
        function_name="_sdpa_bwd_dq_dbias_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_dq",
            (
                int(d["sequence_q"]),
                int(d["head_dimension"]),
                int(d["batch"]) * int(d["heads"]),
            ),
        ),
        dependencies=dependencies,
        num_stages=2,
    )


def _attention_backward_dkdv_stage(
    node: Mapping[str, Any],
) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dk = node["port_tensors"]["dk"]
    dv = node["port_tensors"]["dv"]
    delta_strides = (
        int(d["heads"]) * int(d["sequence_q"]),
        int(d["sequence_q"]),
        1,
    )
    constants = _attention_backward_base(node)
    constants.update(
        {
            "attn_scale": float(d["attn_scale"]),
            "HKV": int(d["key_heads"]),
            "stride_delta_b": delta_strides[0],
            "stride_delta_h": delta_strides[1],
            "stride_delta_m": delta_strides[2],
            "stride_dkb": dk["strides"][0],
            "stride_dkh": dk["strides"][1],
            "stride_dkn": dk["strides"][2],
            "stride_dkd": dk["strides"][3],
            "stride_dvb": dv["strides"][0],
            "stride_dvh": dv["strides"][1],
            "stride_dvn": dv["strides"][2],
            "stride_dvd": dv["strides"][3],
            "Q_PER": int(d["q_per_k"]),
            "BLOCK_D_OUT": _attention_output_dimension_block(
                int(d["head_dimension"])
            ),
        }
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _workspace_pointer("delta_ptr", "delta"),
        _tensor_pointer(node, "dk_ptr", "dk"),
        _tensor_pointer(node, "dv_ptr", "dv"),
    )
    return _make_stage(
        operation="sdpa_backward",
        stage_name="dk_dv",
        function_name="_sdpa_bwd_dkdv_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_dk",
            (
                int(d["sequence_kv"]),
                int(d["head_dimension"]),
                int(d["batch"]) * int(d["key_heads"]),
            ),
        ),
        dependencies=("dq_delta_dbias",),
        num_stages=2,
    )


def _attention_backward_dk_stage(
    node: Mapping[str, Any],
) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dk = node["port_tensors"]["dk"]
    delta_strides = (
        int(d["heads"]) * int(d["sequence_q"]),
        int(d["sequence_q"]),
        1,
    )
    constants = _attention_backward_base(node)
    constants.update(
        {
            "attn_scale": float(d["attn_scale"]),
            "HKV": int(d["key_heads"]),
            "stride_delta_b": delta_strides[0],
            "stride_delta_h": delta_strides[1],
            "stride_delta_m": delta_strides[2],
            "stride_dkb": dk["strides"][0],
            "stride_dkh": dk["strides"][1],
            "stride_dkn": dk["strides"][2],
            "stride_dkd": dk["strides"][3],
            "V_DIM": int(d["value_dimension"]),
            "Q_PER": int(d["q_per_k"]),
            "BLOCK_D_OUT": _attention_output_dimension_block(
                int(d["head_dimension"])
            ),
            "BLOCK_DV": _attention_full_dimension_block(
                int(d["value_dimension"])
            ),
        }
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _workspace_pointer("delta_ptr", "delta"),
        _tensor_pointer(node, "dk_ptr", "dk"),
    )
    return _make_stage(
        operation="sdpa_backward",
        stage_name="dk",
        function_name="_sdpa_bwd_dk_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_dk",
            (
                int(d["sequence_kv"]),
                int(d["head_dimension"]),
                int(d["batch"]) * int(d["key_heads"]),
            ),
        ),
        dependencies=("dq_delta_dbias",),
        num_stages=2,
    )


def _attention_backward_dv_stage(
    node: Mapping[str, Any],
) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dv = node["port_tensors"]["dv"]
    constants = _attention_backward_base(node)
    # The DV-only registry entry neither receives V nor the hidden delta.
    for name in (
        "stride_vb",
        "stride_vh",
        "stride_vn",
        "stride_vd",
        "BLOCK_D_FULL",
    ):
        constants.pop(name)
    constants.update(
        {
            "attn_scale": float(d["attn_scale"]),
            "HKV": int(d["value_heads"]),
            "stride_dvb": dv["strides"][0],
            "stride_dvh": dv["strides"][1],
            "stride_dvn": dv["strides"][2],
            "stride_dvd": dv["strides"][3],
            "V_DIM": int(d["value_dimension"]),
            "Q_PER": int(d["q_per_v"]),
            "BLOCK_D_FULL": _attention_full_dimension_block(
                int(d["head_dimension"])
            ),
            "BLOCK_DV_OUT": _attention_output_dimension_block(
                int(d["value_dimension"])
            ),
        }
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _tensor_pointer(node, "dv_ptr", "dv"),
    )
    return _make_stage(
        operation="sdpa_backward",
        stage_name="dv",
        function_name="_sdpa_bwd_dv_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_dv",
            (
                int(d["sequence_kv"]),
                int(d["value_dimension"]),
                int(d["batch"]) * int(d["value_heads"]),
            ),
        ),
        dependencies=("dk",),
        num_stages=2,
    )


def _attention_backward_plan(node: Mapping[str, Any]) -> NodePlan:
    d = _require_object(node["derived"], "node.derived")
    stages: list[KernelStagePlan] = []
    dq_dependencies: tuple[str, ...] = ()
    if bool(d["has_dbias"]):
        dbias = node["port_tensors"]["dbias"]
        reduces = (
            dbias["dimensions"][0] != d["batch"]
            or dbias["dimensions"][1] != d["heads"]
        )
        if reduces:
            if not _is_contiguous(dbias):
                raise ValueError(
                    "broadcast dbias must be contiguous because the registry "
                    "only declares _zero_contiguous_kernel"
                )
            dbias_elements = _checked_product(
                list(dbias["dimensions"]), "SDPA dbias elements"
            )
            stages.append(
                _make_stage(
                    operation="sdpa_backward",
                    stage_name="zero_dbias",
                    function_name="_zero_contiguous_kernel",
                    pointer_arguments=(_tensor_pointer(node, "ptr", "dbias"),),
                    constants={"n_elements": dbias_elements, "BLOCK": 256},
                    scalar_arguments=(("n_elements", "i32"),),
                    tuning=INTERNAL_TUNING,
                    tuning_key_value=dbias_elements,
                    grid_spec=GridSpec("zero", (dbias_elements,)),
                )
            )
            dq_dependencies = ("zero_dbias",)
    stages.append(_attention_backward_dq_stage(node, dq_dependencies))
    if (
        d["key_heads"] == d["value_heads"]
        and d["head_dimension"] == d["value_dimension"]
    ):
        stages.append(_attention_backward_dkdv_stage(node))
    else:
        stages.append(_attention_backward_dk_stage(node))
        stages.append(_attention_backward_dv_stage(node))

    delta_dimensions = (
        int(d["batch"]),
        int(d["heads"]),
        int(d["sequence_q"]),
    )
    delta_strides = (
        delta_dimensions[1] * delta_dimensions[2],
        delta_dimensions[2],
        1,
    )
    raw_size = 4 * _checked_product(delta_dimensions, "SDPA delta elements")
    workspace_size = (
        _ceil_div(raw_size, WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT
    )
    workspace = WorkspaceTensor(
        name="delta",
        data_type="float32",
        dimensions=delta_dimensions,
        strides=delta_strides,
        offset=0,
        size=raw_size,
    )
    plan = NodePlan(
        "sdpa_backward", tuple(stages), (workspace,), workspace_size
    )
    plan.validate_dependencies()
    return plan


def _fp8_backward_base(
    node: Mapping[str, Any],
) -> dict[str, int | float | bool]:
    d = _require_object(node["derived"], "node.derived")
    q = node["port_tensors"]["q"]
    k = node["port_tensors"]["k"]
    v = node["port_tensors"]["v"]
    o = node["port_tensors"]["o"]
    do = node["port_tensors"]["do"]
    stats = node["port_tensors"]["stats"]
    return {
        "attn_scale": float(d["attn_scale"]),
        "SQ": int(d["sequence_q"]),
        "SKV": int(d["sequence_kv"]),
        "min_diag": int(d["min_diag"]),
        "max_diag": int(d["max_diag"]),
        "stride_qb": q["strides"][0],
        "stride_qh": q["strides"][1],
        "stride_qm": q["strides"][2],
        "stride_qd": q["strides"][3],
        "stride_kb": k["strides"][0],
        "stride_kh": k["strides"][1],
        "stride_kn": k["strides"][2],
        "stride_kd": k["strides"][3],
        "stride_vb": v["strides"][0],
        "stride_vh": v["strides"][1],
        "stride_vn": v["strides"][2],
        "stride_vd": v["strides"][3],
        "stride_ob": o["strides"][0],
        "stride_oh": o["strides"][1],
        "stride_om": o["strides"][2],
        "stride_od": o["strides"][3],
        "stride_dob": do["strides"][0],
        "stride_doh": do["strides"][1],
        "stride_dom": do["strides"][2],
        "stride_dod": do["strides"][3],
        "stride_sb": stats["strides"][0],
        "stride_sh": stats["strides"][1],
        "stride_sm": stats["strides"][2],
        "HEAD_DIM": int(d["head_dimension"]),
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_D": _attention_full_dimension_block(int(d["head_dimension"])),
        "BANDED": bool(d["banded"]),
        # BLOCK_M/BLOCK_N are autotuned and Q/KV/D may have tail tiles.  The
        # common kernel's FULL_BLOCKS path performs unmasked memory accesses,
        # so it is legal only with a proof the planner cannot provide here.
        "FULL_BLOCKS": False,
        "CAUSAL_TOP_LEFT": bool(d["causal_top_left"]),
    }


def _fp8_backward_dq_stage(node: Mapping[str, Any]) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dq = node["port_tensors"]["dq"]
    constants = _fp8_backward_base(node)
    constants.update(
        {
            "HQ": int(d["heads"]),
            "q_per_k": int(d["q_per_k"]),
            "q_per_v": int(d["q_per_v"]),
            "stride_dqb": dq["strides"][0],
            "stride_dqh": dq["strides"][1],
            "stride_dqm": dq["strides"][2],
            "stride_dqd": dq["strides"][3],
        }
    )
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _tensor_pointer(node, "dq_ptr", "dq"),
        _tensor_pointer(node, "amax_dq_ptr", "amax_dq"),
        _tensor_pointer(node, "descale_q_ptr", "descale_q"),
        _tensor_pointer(node, "descale_k_ptr", "descale_k"),
        _tensor_pointer(node, "descale_v_ptr", "descale_v"),
        _tensor_pointer(node, "descale_o_ptr", "descale_o"),
        _tensor_pointer(node, "descale_do_ptr", "descale_do"),
        _tensor_pointer(node, "descale_dp_ptr", "descale_dp"),
        _tensor_pointer(node, "scale_dq_ptr", "scale_dq"),
        _tensor_pointer(node, "scale_dp_ptr", "scale_dp"),
    )
    return _make_stage(
        operation="sdpa_fp8_backward",
        stage_name="dq",
        function_name="_sdpa_fp8_bwd_dq_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=(
            ("attn_scale", "fp32"),
            ("HQ", "i32"),
            ("SQ", "i32"),
            ("SKV", "i32"),
            ("min_diag", "i32"),
            ("max_diag", "i32"),
        ),
        tuning=SDPA_FP8_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_fp8_dq",
            (
                int(d["sequence_q"]),
                int(d["batch"]) * int(d["heads"]),
            ),
        ),
        dependencies=("zero_amax",),
        num_stages=2,
    )


def _fp8_backward_dkdv_stage(node: Mapping[str, Any]) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dk = node["port_tensors"]["dk"]
    dv = node["port_tensors"]["dv"]
    constants = _fp8_backward_base(node)
    constants.update(
        {
            "HKV": int(d["key_heads"]),
            "stride_dkb": dk["strides"][0],
            "stride_dkh": dk["strides"][1],
            "stride_dkn": dk["strides"][2],
            "stride_dkd": dk["strides"][3],
            "stride_dvb": dv["strides"][0],
            "stride_dvh": dv["strides"][1],
            "stride_dvn": dv["strides"][2],
            "stride_dvd": dv["strides"][3],
            "Q_PER": int(d["q_per_k"]),
        }
    )
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _tensor_pointer(node, "dk_ptr", "dk"),
        _tensor_pointer(node, "dv_ptr", "dv"),
        _tensor_pointer(node, "amax_dk_ptr", "amax_dk"),
        _tensor_pointer(node, "amax_dv_ptr", "amax_dv"),
        _tensor_pointer(node, "amax_dp_ptr", "amax_dp"),
        _tensor_pointer(node, "descale_q_ptr", "descale_q"),
        _tensor_pointer(node, "descale_k_ptr", "descale_k"),
        _tensor_pointer(node, "descale_v_ptr", "descale_v"),
        _tensor_pointer(node, "descale_o_ptr", "descale_o"),
        _tensor_pointer(node, "descale_do_ptr", "descale_do"),
        _tensor_pointer(node, "descale_s_ptr", "descale_s"),
        _tensor_pointer(node, "descale_dp_ptr", "descale_dp"),
        _tensor_pointer(node, "scale_s_ptr", "scale_s"),
        _tensor_pointer(node, "scale_dk_ptr", "scale_dk"),
        _tensor_pointer(node, "scale_dv_ptr", "scale_dv"),
        _tensor_pointer(node, "scale_dp_ptr", "scale_dp"),
    )
    return _make_stage(
        operation="sdpa_fp8_backward",
        stage_name="dk_dv",
        function_name="_sdpa_fp8_bwd_dkdv_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_FP8_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_fp8_dkdv",
            (
                int(d["sequence_kv"]),
                int(d["batch"]) * int(d["key_heads"]),
            ),
        ),
        dependencies=("dq",),
        num_stages=2,
    )


def _fp8_backward_plan(node: Mapping[str, Any]) -> NodePlan:
    zero = _make_stage(
        operation="sdpa_fp8_backward",
        stage_name="zero_amax",
        function_name="_zero_sdpa_fp8_bwd_amax_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "amax_dq_ptr", "amax_dq"),
            _tensor_pointer(node, "amax_dk_ptr", "amax_dk"),
            _tensor_pointer(node, "amax_dv_ptr", "amax_dv"),
            _tensor_pointer(node, "amax_dp_ptr", "amax_dp"),
        ),
        constants={},
        tuning=INTERNAL_TUNING,
        tuning_key_value=1,
        grid_spec=GridSpec("fixed", (1, 1, 1)),
    )
    plan = NodePlan(
        "sdpa_fp8_backward",
        (zero, _fp8_backward_dq_stage(node), _fp8_backward_dkdv_stage(node)),
    )
    plan.validate_dependencies()
    return plan


def plan_kernel_stages(node: Mapping[str, Any]) -> NodePlan:
    """Plan all registry-declared stages for a parsed DNN node."""

    operation = node.get("operation")
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported Hygon NN operation {operation!r}")
    if node.get("compute_data_type") != "float32":
        raise ValueError(f"{operation} requires float32 compute_data_type")
    if not isinstance(node.get("derived"), Mapping):
        raise ValueError("node must be produced by compiler_nn.parse_node")
    if operation in CONVOLUTION_OPERATIONS:
        plan = _convolution_plan(node)
    elif operation in NORMALIZATION_OPERATIONS:
        plan = NodePlan(operation, (_normalization_stage(node),))
    elif operation == "sdpa":
        plan = NodePlan(operation, (_attention_forward_stage(node),))
    elif operation == "sdpa_backward":
        plan = _attention_backward_plan(node)
    elif operation == "sdpa_fp8":
        plan = _fp8_forward_plan(node)
    else:
        plan = _fp8_backward_plan(node)
    plan.validate_dependencies()
    return plan


# Provider integration aliases: the multi-stage name is canonical, while the
# shorter aliases make migration from the single-stage planner explicit.
kernel_stage_plan = plan_kernel_stages


def kernel_configurations(
    node: Mapping[str, Any],
) -> tuple[KernelStagePlan, ...]:
    """Return every stage while preserving dependency order."""

    return plan_kernel_stages(node).stages


def kernel_configuration(node: Mapping[str, Any]) -> KernelStagePlan:
    """Return the only stage for a single-stage operation.

    Multi-stage attention must use :func:`plan_kernel_stages`; silently
    dropping zeroing or dependency stages would be a correctness bug.
    """

    plan = plan_kernel_stages(node)
    if len(plan.stages) != 1:
        raise ValueError(
            f"{plan.operation} is multi-stage; use plan_kernel_stages"
        )
    return plan.stages[0]


__all__ = [
    "ATTENTION_OPERATIONS",
    "ALLOW_TF32",
    "CONVOLUTION_OPERATIONS",
    "CONV_FPROP_YOLO_X_P5_GEMM_TUNING",
    "CONV_FPROP_STANDARD_3X3_TUNING",
    "DGRAD_EXACT_3X3_S1_WORKSPACE_CAP",
    "GridSpec",
    "KernelStagePlan",
    "NORMALIZATION_OPERATIONS",
    "NodePlan",
    "OPERATIONS",
    "OPERATION_SCHEMAS",
    "OperationSchema",
    "REGISTRY_FUNCTIONS",
    "SUPPORTED_OPERATIONS",
    "TuningMetadata",
    "WORKSPACE_ALIGNMENT",
    "STRICT_DOT_INPUT_PRECISION",
    "WorkspaceTensor",
    "kernel_configuration",
    "kernel_configurations",
    "kernel_stage_plan",
    "parse_node",
    "plan_kernel_stages",
    "tensor_storage_size",
]
