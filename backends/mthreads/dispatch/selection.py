"""MThreads dispatch selection implementation."""

from __future__ import annotations

from pathlib import Path

from flagdnn_codegen.kernel_registry import select_kernel_candidate

from .attention import (
    _attention_backward_plan,
    _attention_forward_plan,
    _attention_fp8_backward_plan,
    _attention_fp8_forward_plan,
)
from .common import (
    _ATTENTION_SOURCE_RELATIVE_PATH,
    _BINARY_SOURCE_RELATIVE_PATH,
    _COMPOSITE_SOURCE_RELATIVE_PATH,
    _CONV_BIAS_RELU_SOURCE_RELATIVE_PATH,
    _CONVOLUTION_SOURCE_RELATIVE_PATH,
    _IDENTITY_SOURCE_RELATIVE_PATH,
    _LAYOUT_SOURCE_RELATIVE_PATH,
    _MATMUL_SOURCE_RELATIVE_PATH,
    _NORMALIZATION_SOURCE_RELATIVE_PATH,
    _REDUCTION_SOURCE_RELATIVE_PATH,
    _TERNARY_SOURCE_RELATIVE_PATH,
    _UNARY_SOURCE_RELATIVE_PATH,
)
from .composite import _conv_bias_relu_plan
from .convolution import _convolution_plan
from .graph import (
    ParsedAddSquareRequest,
    ParsedAttentionRequest,
    ParsedBatchnormInferenceRequest,
    ParsedBatchnormRequest,
    ParsedBinaryRequest,
    ParsedCompilerRequest,
    ParsedConvBiasReluRequest,
    ParsedConvolutionRequest,
    ParsedLayoutRequest,
    ParsedMatmulRequest,
    ParsedNormalizationRequest,
    ParsedReductionRequest,
    ParsedTernaryRequest,
    ParsedUnaryRequest,
)
from .layout import _layout_plan
from .matmul import _matmul_plan
from .normalization import (
    _batchnorm_inference_plan,
    _batchnorm_plan,
    _normalization_plan,
)
from .pointwise import (
    _add_square_plan,
    _binary_plan,
    _identity_plan,
    _ternary_plan,
    _unary_plan,
)
from .program import ExecutionPlan
from .reduction import _reduction_plan


def _plan(
    request: ParsedCompilerRequest,
    source_sha256: str,
) -> ExecutionPlan:
    from .extended import ParsedExtendedRequest, plan_extended

    if isinstance(request, ParsedExtendedRequest):
        return plan_extended(request, source_sha256)
    if isinstance(request, ParsedBinaryRequest):
        return _binary_plan(request, source_sha256)
    if isinstance(request, ParsedAddSquareRequest):
        return _add_square_plan(request, source_sha256)
    if isinstance(request, ParsedConvBiasReluRequest):
        return _conv_bias_relu_plan(request, source_sha256)
    if isinstance(request, ParsedUnaryRequest):
        if request.operation == "identity":
            return _identity_plan(request, source_sha256)
        return _unary_plan(request, source_sha256)
    if isinstance(request, ParsedTernaryRequest):
        return _ternary_plan(request, source_sha256)
    if isinstance(request, ParsedLayoutRequest):
        return _layout_plan(request, source_sha256)
    if isinstance(request, ParsedReductionRequest):
        return _reduction_plan(request, source_sha256)
    if isinstance(request, ParsedMatmulRequest):
        return _matmul_plan(request, source_sha256)
    if isinstance(request, ParsedConvolutionRequest):
        return _convolution_plan(request, source_sha256)
    if isinstance(request, ParsedNormalizationRequest):
        return _normalization_plan(request, source_sha256)
    if isinstance(request, ParsedBatchnormRequest):
        return _batchnorm_plan(request, source_sha256)
    if isinstance(request, ParsedBatchnormInferenceRequest):
        return _batchnorm_inference_plan(request, source_sha256)
    if isinstance(request, ParsedAttentionRequest):
        if request.operation == "sdpa":
            return _attention_forward_plan(request, source_sha256)
        if request.operation == "sdpa_backward":
            return _attention_backward_plan(request, source_sha256)
        if request.operation == "sdpa_fp8":
            return _attention_fp8_forward_plan(request, source_sha256)
        if request.operation == "sdpa_fp8_backward":
            return _attention_fp8_backward_plan(request, source_sha256)
        raise ValueError("unknown MThreads Attention operation")
    raise TypeError("unknown mthreads compiler request family")


def select_source(request):
    from .extended import ParsedExtendedRequest

    if isinstance(request, ParsedExtendedRequest):
        candidate = select_kernel_candidate(
            "mthreads",
            (
                "matmul_fp8"
                if request.operation == "matmul"
                else request.operation
            ),
        )
        if (
            candidate.ownership != "common"
            or Path(candidate.source).name != candidate.source
        ):
            raise ValueError("extended kernel registry source is invalid")
        return candidate, "kernels/" + candidate.source
    candidate = select_kernel_candidate("mthreads", request.operation)
    platform_kernel = (
        isinstance(
            request,
            (
                ParsedBinaryRequest,
                ParsedAddSquareRequest,
                ParsedConvBiasReluRequest,
                ParsedBatchnormInferenceRequest,
            ),
        )
        or isinstance(
            request,
            (
                ParsedConvolutionRequest,
                ParsedUnaryRequest,
                ParsedMatmulRequest,
            ),
        )
        or (
            isinstance(request, ParsedLayoutRequest)
            and request.operation in {"reshape", "slice", "transpose"}
        )
    )
    if candidate.ownership not in {"common", "platform"} or (
        candidate.ownership == "platform" and not platform_kernel
    ):
        raise ValueError(
            "mthreads compiler selected an unsupported kernel owner"
        )
    if isinstance(request, ParsedBinaryRequest):
        expected_source = "binary.py"
        source_relative_path = _BINARY_SOURCE_RELATIVE_PATH
        required_functions = {
            "binary_contiguous_kernel",
            "binary_strided_kernel",
        }
    elif isinstance(request, ParsedAddSquareRequest):
        expected_source = "composite.py"
        source_relative_path = _COMPOSITE_SOURCE_RELATIVE_PATH
        required_functions = {"add_square_tensor_kernel"}
    elif isinstance(request, ParsedConvBiasReluRequest):
        expected_source = "conv_bias_relu.py"
        source_relative_path = _CONV_BIAS_RELU_SOURCE_RELATIVE_PATH
        required_functions = {"conv_bias_relu_2d_kernel"}
    elif isinstance(request, ParsedUnaryRequest):
        if request.operation == "identity":
            expected_source = "identity.py"
            source_relative_path = _IDENTITY_SOURCE_RELATIVE_PATH
            required_functions = {
                "identity_contiguous_packed_kernel",
                "identity_contiguous_kernel",
                "identity_strided_kernel",
            }
        else:
            expected_source = "unary.py"
            source_relative_path = _UNARY_SOURCE_RELATIVE_PATH
            required_functions = {
                "unary_pointwise_contiguous_kernel",
                "unary_pointwise_strided_kernel",
            }
    elif isinstance(request, ParsedTernaryRequest):
        expected_source = "ternary.py"
        source_relative_path = _TERNARY_SOURCE_RELATIVE_PATH
        required_functions = {
            "binary_select_tensor_kernel",
            "binary_select_strided_kernel",
        }
    elif isinstance(request, ParsedLayoutRequest):
        expected_source = "layout.py"
        source_relative_path = _LAYOUT_SOURCE_RELATIVE_PATH
        required_functions = (
            {"reshape_contiguous_kernel", "layout_copy_kernel"}
            if request.operation == "reshape"
            else (
                {"slice_copy_kernel", "layout_copy_kernel"}
                if request.operation == "slice"
                else (
                    {"transpose_physical_copy_kernel", "layout_copy_kernel"}
                    if request.operation == "transpose"
                    else {"layout_copy_kernel"}
                )
            )
        )
    elif isinstance(request, ParsedReductionRequest):
        expected_source = "reduction.py"
        source_relative_path = _REDUCTION_SOURCE_RELATIVE_PATH
        required_functions = {
            "reduction_2d_kernel",
            "reduction_3d_kernel",
            "reduction_strided_kernel",
        }
    elif isinstance(request, ParsedMatmulRequest):
        expected_source = "matmul.py"
        source_relative_path = _MATMUL_SOURCE_RELATIVE_PATH
        required_functions = {
            "matmul_descriptor_kernel",
            "_matmul_tle_consumer",
            "_matmul_tle_producer",
            "matmul_tle_kernel",
            "matmul_strided_kernel",
        }
    elif isinstance(request, ParsedConvolutionRequest):
        expected_source = "convolution.py"
        source_relative_path = _CONVOLUTION_SOURCE_RELATIVE_PATH
        if request.operation == "convolution_dgrad":
            required_functions = {"conv_dgrad_nd_kernel"}
        elif request.operation == "convolution_wgrad":
            required_functions = {
                "conv_wgrad_nd_kernel",
                "_conv_wgrad2d_p5_pack_image_kernel",
                "_conv_wgrad2d_p5_mm_kernel",
            }
        else:
            required_functions = {
                "conv1d_gemm_kernel",
                "conv2d_spatial_nchw_kernel",
                "conv3d_spatial_ncdhw_m_kernel",
                "_conv_fprop2d_im2col_kernel",
                "_conv_fprop2d_im2col_mm_kernel",
                "conv_dgrad_nd_kernel",
                "conv_wgrad_nd_kernel",
            }
    elif isinstance(request, ParsedNormalizationRequest):
        expected_source = "normalization.py"
        source_relative_path = _NORMALIZATION_SOURCE_RELATIVE_PATH
        required_functions = {
            (
                "layer_norm_kernel"
                if request.operation == "layernorm"
                else "rms_norm_kernel"
            )
        }
    elif isinstance(request, ParsedBatchnormRequest):
        expected_source = "normalization.py"
        source_relative_path = _NORMALIZATION_SOURCE_RELATIVE_PATH
        required_functions = {
            "batch_norm_nchw_kernel",
            "batch_norm_kernel",
        }
    elif isinstance(request, ParsedBatchnormInferenceRequest):
        expected_source = "normalization.py"
        source_relative_path = _NORMALIZATION_SOURCE_RELATIVE_PATH
        required_functions = {
            "batch_norm_inference_nchw_kernel",
            "batch_norm_inference_kernel",
        }
    elif isinstance(request, ParsedAttentionRequest):
        expected_source = "attention.py"
        source_relative_path = _ATTENTION_SOURCE_RELATIVE_PATH
        required_functions = {
            "sdpa": {"_sdpa_fwd_kernel"},
            "sdpa_backward": {
                "_zero_contiguous_kernel",
                "_sdpa_bwd_dq_dbias_kernel",
                "_sdpa_bwd_dkdv_kernel",
                "_sdpa_bwd_dk_kernel",
                "_sdpa_bwd_dv_kernel",
            },
            "sdpa_fp8": {
                "_zero_sdpa_fp8_fwd_amax_kernel",
                "_sdpa_fp8_fwd_kernel",
            },
            "sdpa_fp8_backward": {
                "_zero_sdpa_fp8_bwd_amax_kernel",
                "_sdpa_fp8_bwd_dq_kernel",
                "_sdpa_fp8_bwd_dkdv_kernel",
            },
        }[request.operation]
    else:
        raise TypeError("unknown mthreads compiler request family")
    if required_functions.difference(candidate.functions):
        raise ValueError("kernel registry entry is incomplete")
    if (
        Path(candidate.source).is_absolute()
        or ".." in Path(candidate.source).parts
        or candidate.source != expected_source
    ):
        raise ValueError("common kernel source path is unsafe")
    return candidate, source_relative_path
