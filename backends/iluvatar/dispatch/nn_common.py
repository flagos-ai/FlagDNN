# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / nn_common implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping
from typing import Sequence
import math

ILUVATAR_WARP_SIZE = 64


MAX_WORKGROUP_SIZE = 1024


WORKSPACE_ALIGNMENT = 256


MAX_RANK = 8


MAX_I32 = 2**31 - 1


MIN_I32 = -(2**31)


MAX_U32 = 2**32 - 1


MAX_I64 = 2**63 - 1


MAX_FLOAT32 = 3.4028234663852886e38


UNBOUNDED_DIAGONAL = 1 << 30


# Provider integration must preserve this policy. The installed IX CoreX backend
# defaults tl.dot to IEEE and the convolution entries also hard-code IEEE.
STRICT_DOT_INPUT_PRECISION = "ieee"


ALLOW_TF32 = False


POINTER_TYPES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    # CoreX cannot masked-load fp8e4b15 with a zero fill value. Keep the API
    # encoding as raw bytes and bitcast inside the attention kernels.
    "fp8_e4m3": "*i8",
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


# Source-of-truth mirror of the registry-selected common or Iluvatar provider.
# Planning code may not select outside these declared entry points.
_CONV_FPROP_FUNCTIONS = (
    "conv1d_gemm_kernel",
    "conv2d_1x1_nchw_kernel",
    "conv_fprop_2d_im2col_kernel",
    "conv_fprop_2d_packed_matmul_kernel",
    "conv2d_3x3_nchw_pad1_kernel",
    "conv2d_fp32_ml_stem_kernel",
    "conv2d_spatial_nchw_kernel",
    "conv3d_spatial_ncdhw_m_kernel",
)


_CONV_DGRAD_FUNCTIONS = (
    "conv_dgrad_2d_1x1_kernel",
    "conv_dgrad_2d_p5_splitk_kernel",
    "conv_dgrad_cast_contiguous_kernel",
    "conv_dgrad_pack_weight_3x3_kernel",
    "conv_dgrad_2d_packed_parity_kernel",
    "conv_dgrad_zero_kernel",
    "conv_dgrad_2d_scatter_kernel",
    "conv_dgrad_2d_stride2_kernel",
    "conv_dgrad_nd_kernel",
)


_CONV_WGRAD_FUNCTIONS = (
    "conv_dgrad_cast_contiguous_kernel",
    "conv_wgrad_2d_p5_pack_kernel",
    "conv_wgrad_2d_p5_matmul_kernel",
    "conv_wgrad_2d_col_split_kernel",
    "conv_wgrad_2d_col_reduce_kernel",
    "conv_wgrad_2d_pack_image_kernel",
    "conv_wgrad_2d_batched_matmul_kernel",
    "conv_wgrad_2d_batched_wide_kernel",
    "conv_wgrad_2d_batched_reduce_kernel",
    "conv_wgrad_1d_3tap_split_kernel",
    "conv_wgrad_1d_3tap_reduce_kernel",
    "conv_wgrad_2d_1x1_split_kernel",
    "conv_wgrad_2d_1x1_reduce_kernel",
    "conv_wgrad_nd_kernel",
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
        "_convert_fp8_to_fp16_4d_kernel",
        "_zero_sdpa_fp8_fwd_amax_kernel",
        "_sdpa_fp8_fwd_kernel",
    ),
    "sdpa_fp8_backward": (
        "_convert_fp8_to_fp16_4d_kernel",
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
        "BLOCK_K",
        "GROUP_M",
    ),
)


CONV_DGRAD_TUNING = TuningMetadata(
    "convolution.yaml",
    "conv_dgrad",
    "n_outputs",
    "convolution",
    meta_keys=(
        "BLOCK_M",
        "BLOCK_CI",
        "BLOCK_K",
        "GROUP_M",
    ),
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
    ),
)


CONV_TUNING_BY_OPERATION = {
    "conv2d_fprop": CONV_FPROP_TUNING,
    "convolution_fprop": CONV_FPROP_TUNING,
    "convolution_dgrad": CONV_DGRAD_TUNING,
    "convolution_wgrad": CONV_WGRAD_TUNING,
}


UNTUNED_TUNING = TuningMetadata("", "", "", "")


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
    "align64",
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
        raise ValueError("Iluvatar launch grid dimensions must fit uint32")


def _validate_launch(num_warps: int, num_stages: int) -> None:
    if (
        isinstance(num_warps, bool)
        or not isinstance(num_warps, int)
        or num_warps <= 0
        or (num_warps & (num_warps - 1)) != 0
        or num_warps * ILUVATAR_WARP_SIZE > MAX_WORKGROUP_SIZE
    ):
        raise ValueError(
            "Iluvatar num_warps must be a positive power of two and "
            "num_warps * 64 <= 1024"
        )
    if (
        isinstance(num_stages, bool)
        or not isinstance(num_stages, int)
        or not 1 <= num_stages <= 32
    ):
        raise ValueError("Iluvatar num_stages must be in [1, 32]")


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
                batch * _ceil_div(channels, block_c) * _ceil_div(spatial, block_s),
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
        elif self.kind == "conv_dgrad_scatter":
            rows, columns, groups = self.extents
            result = (
                _ceil_div(rows, _meta_int(constants, "BLOCK_M"))
                * _ceil_div(columns, _meta_int(constants, "BLOCK_N")),
                groups,
                1,
            )
        elif self.kind == "conv_dgrad_stride2_parity":
            batch, height, width, channels, groups = self.extents
            parity_h = constants.get("PARITY_H")
            parity_w = constants.get("PARITY_W")
            if (
                isinstance(parity_h, bool)
                or not isinstance(parity_h, int)
                or parity_h not in (0, 1)
                or isinstance(parity_w, bool)
                or not isinstance(parity_w, int)
                or parity_w not in (0, 1)
            ):
                raise ValueError("DGrad parity constants must be 0 or 1")
            parity_height = (height + 1 - parity_h) // 2
            parity_width = (width + 1 - parity_w) // 2
            rows = batch * parity_height * parity_width
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
                _ceil_div(value_dimension, _meta_int(constants, "BLOCK_DV_OUT")),
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
            raise ValueError(f"unknown Iluvatar NN grid kind {self.kind!r}")
        if len(result) != 3:
            raise ValueError("Iluvatar grid must have three dimensions")
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
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"tuning META.{name} must be a positive integer")
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
        raise ValueError(f"parameters.{name} must be in [{minimum}, {maximum}]")
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
            f"parameters.{name} values must be integers in " f"[{minimum}, {maximum}]"
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
    dimensions = _require_sequence(tensor.get("dimensions"), f"{name}.dimensions")
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
        raise ValueError(f"node {direction} port must be named {expected_name!r}")
    optional = port.get("optional", False)
    if not isinstance(optional, bool) or optional:
        raise ValueError("present Iluvatar NN tensor ports must be non-optional")
    uid = port.get("uid")
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError(f"node {direction} UID is invalid")
    try:
        tensor = tensor_registry[uid]
    except KeyError as error:
        raise ValueError(f"node references unknown tensor UID {uid}") from error
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
        if "has_dbias" in parameters and _require_flag(parameters, "has_dbias"):
            raise ValueError(f"{operation} does not support bias gradients")
    return inputs, outputs


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
        raise ValueError(f"parameters.{name} is inconsistent with tensor metadata")
    return result


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
            raise ValueError(f"pointer argument {argument_name!r} is not a pointer")
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
                    "represented by the Iluvatar scalar_i32 artifact ABI"
                )
            runtime_value: int | float = int(value)
            layout_kind = "scalar_i32"
        elif token == "fp32":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"runtime scalar {argument_name!r} must be numeric")
            runtime_value = float(value)
            if not math.isfinite(runtime_value) or abs(runtime_value) > MAX_FLOAT32:
                raise ValueError(
                    f"runtime scalar {argument_name!r}={value!r} cannot be "
                    "represented by the Iluvatar scalar_f32 artifact ABI"
                )
            layout_kind = "scalar_f32"
        else:
            raise ValueError(
                f"unsupported runtime scalar token {token!r} for " f"{argument_name!r}"
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
        meta_keys=tuple(name for name in tuning.meta_keys if name in specialized),
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


def _contiguous_strides(dimensions: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(dimensions):
        result.append(stride)
        stride *= int(dimension)
    return tuple(reversed(result))


def _aligned_size(size: int) -> int:
    return _ceil_div(size, WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT


def _workspace_tensor(
    name: str,
    data_type: str,
    dimensions: Sequence[int],
    *,
    offset: int = 0,
) -> WorkspaceTensor:
    typed_dimensions = tuple(int(value) for value in dimensions)
    elements = _checked_product(list(typed_dimensions), f"{name} workspace elements")
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


LOG2_E = 1.4426950408889634
