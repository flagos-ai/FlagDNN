/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "convolution_reference.hpp"

#include "activation_layout.hpp"
#include "hip_driver.hpp"
#include "tensor_io.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::validation::hygon {
namespace {

std::string status_name(hipdnnStatus_t status) {
  switch (status) {
  case HIPDNN_STATUS_SUCCESS:
    return "HIPDNN_STATUS_SUCCESS";
  case HIPDNN_STATUS_NOT_INITIALIZED:
    return "HIPDNN_STATUS_NOT_INITIALIZED";
  case HIPDNN_STATUS_ALLOC_FAILED:
    return "HIPDNN_STATUS_ALLOC_FAILED";
  case HIPDNN_STATUS_BAD_PARAM:
    return "HIPDNN_STATUS_BAD_PARAM";
  case HIPDNN_STATUS_INTERNAL_ERROR:
    return "HIPDNN_STATUS_INTERNAL_ERROR";
  case HIPDNN_STATUS_INVALID_VALUE:
    return "HIPDNN_STATUS_INVALID_VALUE";
  case HIPDNN_STATUS_ARCH_MISMATCH:
    return "HIPDNN_STATUS_ARCH_MISMATCH";
  case HIPDNN_STATUS_MAPPING_ERROR:
    return "HIPDNN_STATUS_MAPPING_ERROR";
  case HIPDNN_STATUS_EXECUTION_FAILED:
    return "HIPDNN_STATUS_EXECUTION_FAILED";
  case HIPDNN_STATUS_NOT_SUPPORTED:
    return "HIPDNN_STATUS_NOT_SUPPORTED";
  case HIPDNN_STATUS_LICENSE_ERROR:
    return "HIPDNN_STATUS_LICENSE_ERROR";
  case HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
    return "HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
  case HIPDNN_STATUS_RUNTIME_IN_PROGRESS:
    return "HIPDNN_STATUS_RUNTIME_IN_PROGRESS";
  case HIPDNN_STATUS_RUNTIME_FP_OVERFLOW:
    return "HIPDNN_STATUS_RUNTIME_FP_OVERFLOW";
  case HIPDNN_STATUS_VERSION_MISMATCH:
    return "HIPDNN_STATUS_VERSION_MISMATCH";
  }
  return "HIPDNN_STATUS_UNKNOWN";
}

void check_hipdnn(hipdnnStatus_t status, std::string_view stage) {
  check_hipdnn_status(status, stage);
}

bool is_setup_capability_status(hipdnnStatus_t status) noexcept {
  return hipdnn_status_is_capability(status);
}

bool is_execute_capability_status(hipdnnStatus_t status) noexcept {
  return is_setup_capability_status(status);
}

bool is_hip_capability_status(hipError_t status) noexcept {
  (void)status;
  return false;
}

hipdnnDataType_t hipdnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return HIPDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return HIPDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return HIPDNN_DATA_BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("data type has no hipDNN convolution mapping");
}

hipdnnConvolutionMode_t hipdnn_convolution_mode(HipdnnConvolutionMode mode) {
  switch (mode) {
  case HipdnnConvolutionMode::kCrossCorrelation:
    return HIPDNN_CROSS_CORRELATION;
  case HipdnnConvolutionMode::kConvolution:
    return HIPDNN_CONVOLUTION;
  }
  throw std::invalid_argument("invalid hipDNN convolution mode");
}

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t &result) noexcept {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

std::vector<std::int64_t>
contiguous_strides(std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error("convolution contiguous strides overflow");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

std::vector<std::int64_t>
channels_last_strides(std::span<const std::int64_t> dimensions) {
  if (dimensions.size() < 3 || dimensions.size() > 5) {
    throw std::invalid_argument("channels-last convolution rank is invalid");
  }
  std::vector<std::int64_t> result(dimensions.size());
  result[1] = 1;
  std::int64_t stride = dimensions[1];
  for (std::size_t axis = dimensions.size(); axis != 2; --axis) {
    const std::size_t current = axis - 1;
    result[current] = stride;
    if (dimensions[current] <= 0 ||
        stride >
            std::numeric_limits<std::int64_t>::max() / dimensions[current]) {
      throw std::overflow_error("channels-last convolution strides overflow");
    }
    stride *= dimensions[current];
  }
  result[0] = stride;
  return result;
}

std::optional<hipdnnTensorFormat_t>
canonical_format(const ReferenceTensor &tensor) {
  try {
    if (tensor.strides == contiguous_strides(tensor.dimensions)) {
      return HIPDNN_TENSOR_NCHW;
    }
    if (tensor.strides == channels_last_strides(tensor.dimensions)) {
      return HIPDNN_TENSOR_NHWC;
    }
  } catch (const std::exception &) {
    return std::nullopt;
  }
  return std::nullopt;
}

bool metadata_fits_int(const ReferenceTensor &tensor) noexcept {
  if (tensor.dimensions.size() < 3 || tensor.dimensions.size() > 5 ||
      tensor.dimensions.size() != tensor.strides.size()) {
    return false;
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0 ||
        tensor.dimensions[axis] > std::numeric_limits<int>::max() ||
        tensor.strides[axis] > std::numeric_limits<int>::max()) {
      return false;
    }
  }
  return true;
}

bool metadata_contract_valid(const ReferenceTensor &tensor) noexcept {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    return false;
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      return false;
    }
  }
  return true;
}

std::vector<int> checked_ints(std::span<const std::int64_t> values,
                              std::string_view role, bool allow_zero) {
  std::vector<int> result;
  result.reserve(values.size());
  for (std::int64_t value : values) {
    if (value > std::numeric_limits<int>::max() ||
        (allow_zero ? value < 0 : value <= 0)) {
      throw std::invalid_argument(std::string(role) +
                                  " does not fit hipDNN int metadata");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

std::int64_t output_dimension(std::int64_t input, std::int64_t filter,
                              std::int64_t padding, std::int64_t stride,
                              std::int64_t dilation) {
  return (input + 2 * padding - dilation * (filter - 1) - 1) / stride + 1;
}

std::size_t tensor_bytes(const ReferenceTensor &tensor) {
  const std::size_t elements = tensor_io::storage_element_count(tensor);
  const std::size_t element_size = tensor_io::data_type_size(tensor.data_type);
  std::size_t bytes = 0;
  if (!checked_multiply(elements, element_size, bytes)) {
    throw std::overflow_error("hipDNN convolution tensor storage overflows");
  }
  return bytes;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("workspace alignment must be a power of two");
  }
  const std::size_t mask = alignment - 1;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    throw std::overflow_error("hipDNN convolution workspace size overflows");
  }
  return (value + mask) & ~mask;
}

void *binding_pointer(std::span<const flagdnnBinding_t> bindings,
                      const ReferenceTensor &tensor) {
  const auto found = std::find_if(bindings.begin(), bindings.end(),
                                  [&](const flagdnnBinding_t &binding) {
                                    return binding.uid == tensor.uid;
                                  });
  if (found == bindings.end() || found->device_pointer == nullptr) {
    throw std::invalid_argument("hipDNN convolution binding is missing");
  }
  return static_cast<void *>(
      static_cast<std::uint8_t *>(found->device_pointer) +
      tensor.binding_byte_offset);
}

HipdnnCapability
structural_capability(const HipdnnConvolutionOperation &operation,
                      std::span<const ReferenceTensor> tensors) {
  if (operation.kind == HipdnnConvolutionKind::kUnavailable) {
    return HipdnnCapability::vendor_unsupported(
        operation.unavailable_reason.empty()
            ? "no exact hipDNN convolution primitive"
            : operation.unavailable_reason);
  }
  const std::size_t expected =
      operation.kind == HipdnnConvolutionKind::kConvBiasRelu ? 4 : 3;
  if (tensors.size() != expected) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN convolution semantic tensor count is invalid");
  }
  const ReferenceTensor &x = tensors[0];
  const ReferenceTensor &w = tensors[1];
  const ReferenceTensor &y = tensors[2];
  if (x.uid == w.uid || x.uid == y.uid || w.uid == y.uid ||
      (expected == 4 && (tensors[3].uid == x.uid || tensors[3].uid == w.uid ||
                         tensors[3].uid == y.uid))) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN convolution tensor UIDs must be unique");
  }
  if (x.data_type != w.data_type || x.data_type != y.data_type ||
      (expected == 4 && tensors[3].data_type != x.data_type)) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN convolution tensor data types must match");
  }
  if (x.data_type != FLAGDNN_DATA_FLOAT32 &&
      x.data_type != FLAGDNN_DATA_FLOAT16 &&
      x.data_type != FLAGDNN_DATA_BFLOAT16) {
    return HipdnnCapability::vendor_unsupported(
        "convolution dtype has no hipDNN primitive mapping");
  }
  if (operation.kind == HipdnnConvolutionKind::kConvBiasRelu &&
      x.data_type == FLAGDNN_DATA_BFLOAT16) {
    return HipdnnCapability::vendor_unsupported(
        "BF16 ConvBiasRelu is outside the validated DTK hipDNN public "
        "OpTensor bias-add capability");
  }
  for (const ReferenceTensor &tensor : tensors) {
    if (!metadata_contract_valid(tensor)) {
      return HipdnnCapability::invalid_adapter_contract(
          "convolution tensor dimensions/strides metadata is invalid");
    }
  }
  if (x.dimensions.size() != w.dimensions.size() ||
      x.dimensions.size() != y.dimensions.size()) {
    return HipdnnCapability::invalid_adapter_contract(
        "convolution tensor ranks must match");
  }
  if (!metadata_fits_int(x) || !metadata_fits_int(w) || !metadata_fits_int(y) ||
      (expected == 4 && !metadata_fits_int(tensors[3]))) {
    return HipdnnCapability::vendor_unsupported(
        "convolution rank/shape/stride metadata does not fit hipDNN");
  }
  if (x.dimensions.size() != 4) {
    return HipdnnCapability::vendor_unsupported(
        "DTK hipDNN public convolution execute is validated only for 2D "
        "tensors");
  }
  const std::optional<hipdnnTensorFormat_t> x_format = canonical_format(x);
  const std::optional<hipdnnTensorFormat_t> w_format = canonical_format(w);
  const std::optional<hipdnnTensorFormat_t> y_format = canonical_format(y);
  if (!x_format.has_value() || !w_format.has_value() || !y_format.has_value()) {
    return HipdnnCapability::vendor_unsupported(
        "hipDNN convolution is validated only for canonical channel-first or "
        "channels-last layouts");
  }
  if (*x_format != *w_format || *x_format != *y_format) {
    return HipdnnCapability::vendor_unsupported(
        "DTK hipDNN public convolution execute is not validated for mixed "
        "tensor layouts");
  }
  const std::size_t element_size = tensor_io::data_type_size(x.data_type);
  for (const ReferenceTensor &tensor : tensors) {
    if (tensor.binding_byte_offset % element_size != 0) {
      return HipdnnCapability::invalid_adapter_contract(
          "convolution binding byte offset is not element aligned");
    }
  }

  const std::size_t spatial_rank = x.dimensions.size() - 2;
  if (operation.padding.size() != spatial_rank ||
      operation.stride.size() != spatial_rank ||
      operation.dilation.size() != spatial_rank || operation.groups <= 0) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN convolution attributes have an invalid rank");
  }
  if (operation.groups > std::numeric_limits<int>::max()) {
    return HipdnnCapability::vendor_unsupported(
        "convolution group count does not fit hipDNN metadata");
  }
  if (x.dimensions[1] % operation.groups != 0 ||
      w.dimensions[0] % operation.groups != 0 ||
      w.dimensions[1] != x.dimensions[1] / operation.groups ||
      y.dimensions[0] != x.dimensions[0] ||
      y.dimensions[1] != w.dimensions[0]) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN convolution channel/group metadata is invalid");
  }
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    if (operation.padding[axis] < 0 || operation.stride[axis] <= 0 ||
        operation.dilation[axis] <= 0) {
      return HipdnnCapability::invalid_adapter_contract(
          "hipDNN convolution spatial attributes are invalid");
    }
    if (operation.padding[axis] > std::numeric_limits<int>::max() ||
        operation.stride[axis] > std::numeric_limits<int>::max() ||
        operation.dilation[axis] > std::numeric_limits<int>::max()) {
      return HipdnnCapability::vendor_unsupported(
          "convolution spatial attributes do not fit hipDNN metadata");
    }
    if (y.dimensions[axis + 2] !=
        output_dimension(x.dimensions[axis + 2], w.dimensions[axis + 2],
                         operation.padding[axis], operation.stride[axis],
                         operation.dilation[axis])) {
      return HipdnnCapability::invalid_adapter_contract(
          "hipDNN convolution output shape is not exact");
    }
  }

  if (operation.kind == HipdnnConvolutionKind::kConvBiasRelu) {
    const ReferenceTensor &bias = tensors[3];
    if (x.dimensions.size() != 4 || !metadata_fits_int(bias) ||
        bias.dimensions.size() != 4 || !canonical_format(bias).has_value() ||
        bias.dimensions[0] != 1 || bias.dimensions[1] != y.dimensions[1] ||
        bias.dimensions[2] != 1 || bias.dimensions[3] != 1) {
      return HipdnnCapability::vendor_unsupported(
          "ConvBiasRelu bias/layout is not exactly representable by hipDNN "
          "primitives");
    }
  }
  return {};
}

int convolution_algorithm_count(HipdnnConvolutionKind kind) {
  switch (kind) {
  case HipdnnConvolutionKind::kFprop:
  case HipdnnConvolutionKind::kConvBiasRelu:
    return static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_COUNT);
  case HipdnnConvolutionKind::kDgrad:
    return static_cast<int>(HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);
  case HipdnnConvolutionKind::kWgrad:
    return static_cast<int>(HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
  case HipdnnConvolutionKind::kUnavailable:
    break;
  }
  throw std::invalid_argument(
      "unavailable convolution kind has no hipDNN algorithm set");
}

int correctness_oracle_algorithm(HipdnnConvolutionKind kind) {
  switch (kind) {
  case HipdnnConvolutionKind::kFprop:
  case HipdnnConvolutionKind::kConvBiasRelu:
    return static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
  case HipdnnConvolutionKind::kDgrad:
    return static_cast<int>(HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1);
  case HipdnnConvolutionKind::kWgrad:
    return static_cast<int>(HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
  case HipdnnConvolutionKind::kUnavailable:
    break;
  }
  throw std::invalid_argument(
      "unavailable convolution kind has no correctness oracle algorithm");
}

} // namespace

HipdnnConvolutionOperation make_hipdnn_convolution_operation(
    HipdnnConvolutionKind kind, std::vector<std::int64_t> pre_padding,
    std::vector<std::int64_t> post_padding, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> dilation, std::int64_t groups,
    HipdnnConvolutionMode mode) {
  if (pre_padding != post_padding) {
    return make_hipdnn_convolution_unavailable(
        "hipDNN public convolution descriptor cannot represent asymmetric "
        "pre/post padding exactly");
  }
  HipdnnConvolutionOperation result;
  result.kind = kind;
  result.padding = std::move(pre_padding);
  result.stride = std::move(stride);
  result.dilation = std::move(dilation);
  result.groups = groups;
  result.mode = mode;
  return result;
}

HipdnnConvolutionOperation
make_hipdnn_convolution_unavailable(std::string reason) {
  HipdnnConvolutionOperation result;
  result.unavailable_reason = std::move(reason);
  return result;
}

HipdnnCapability
hipdnn_convolution_capability(const HipdnnConvolutionOperation &operation,
                              std::span<const ReferenceTensor> tensors) {
  return structural_capability(operation, tensors);
}

std::string_view
hipdnn_convolution_kind_name(HipdnnConvolutionKind kind) noexcept {
  switch (kind) {
  case HipdnnConvolutionKind::kFprop:
    return "conv_fprop";
  case HipdnnConvolutionKind::kDgrad:
    return "conv_dgrad";
  case HipdnnConvolutionKind::kWgrad:
    return "conv_wgrad";
  case HipdnnConvolutionKind::kConvBiasRelu:
    return "conv_bias_relu";
  case HipdnnConvolutionKind::kUnavailable:
    return "unavailable";
  }
  return "unknown";
}

std::string_view hipdnn_convolution_algorithm_policy_name(
    HipdnnConvolutionAlgorithmPolicy policy) noexcept {
  switch (policy) {
  case HipdnnConvolutionAlgorithmPolicy::kCorrectnessOracle:
    return "correctness_oracle";
  case HipdnnConvolutionAlgorithmPolicy::kPerformance:
    return "performance";
  }
  return "unknown";
}

std::vector<int>
hipdnn_convolution_algorithm_order(HipdnnConvolutionKind kind,
                                   HipdnnConvolutionAlgorithmPolicy policy,
                                   std::span<const int> heuristic_values) {
  const int count = convolution_algorithm_count(kind);
  const int oracle = correctness_oracle_algorithm(kind);
  if (policy == HipdnnConvolutionAlgorithmPolicy::kCorrectnessOracle) {
    return {oracle};
  }
  if (policy != HipdnnConvolutionAlgorithmPolicy::kPerformance) {
    throw std::invalid_argument("invalid hipDNN convolution algorithm policy");
  }

  std::vector<int> result;
  result.reserve(static_cast<std::size_t>(count));
  const auto append = [&](int value) {
    if (value < 0 || value >= count) {
      throw std::invalid_argument(
          "hipDNN convolution heuristic returned an invalid algorithm");
    }
    if (value == oracle) {
      return;
    }
    if (std::find(result.begin(), result.end(), value) == result.end()) {
      result.push_back(value);
    }
  };
  for (const int value : heuristic_values) {
    append(value);
  }
  for (int value = 0; value < count; ++value) {
    append(value);
  }
  result.push_back(oracle);
  return result;
}

std::string_view hipdnn_convolution_algorithm_name(HipdnnConvolutionKind kind,
                                                   int algorithm) noexcept {
  if (kind == HipdnnConvolutionKind::kFprop ||
      kind == HipdnnConvolutionKind::kConvBiasRelu) {
    switch (static_cast<hipdnnConvolutionFwdAlgo_t>(algorithm)) {
    case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
      return "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
    case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
      return "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
    case HIPDNN_CONVOLUTION_FWD_ALGO_GEMM:
      return "HIPDNN_CONVOLUTION_FWD_ALGO_GEMM";
    case HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT:
      return "HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT";
    case HIPDNN_CONVOLUTION_FWD_ALGO_FFT:
      return "HIPDNN_CONVOLUTION_FWD_ALGO_FFT";
    case HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
      return "HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
    case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
      return "HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
    case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      return "HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
    case HIPDNN_CONVOLUTION_FWD_ALGO_COUNT:
      break;
    }
  } else if (kind == HipdnnConvolutionKind::kDgrad) {
    switch (static_cast<hipdnnConvolutionBwdDataAlgo_t>(algorithm)) {
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0:
      return "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0";
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1:
      return "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1";
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
      return "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
      return "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
      return "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
      return "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
      break;
    }
  } else if (kind == HipdnnConvolutionKind::kWgrad) {
    switch (static_cast<hipdnnConvolutionBwdFilterAlgo_t>(algorithm)) {
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
      return "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0";
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
      return "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1";
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
      return "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT";
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
      return "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3";
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
      return "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD";
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
      return "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED";
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
      return "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING";
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
      break;
    }
  }
  return "HIPDNN_CONVOLUTION_ALGO_UNKNOWN";
}

bool hipdnn_convolution_heuristic_allows_enum_fallback(
    hipdnnStatus_t status) noexcept {
  return status == HIPDNN_STATUS_NOT_SUPPORTED ||
         status == HIPDNN_STATUS_EXECUTION_FAILED;
}

class HipdnnConvolutionPlan::Impl final {
public:
  Impl(HipdnnConvolutionOperation operation,
       std::vector<ReferenceTensor> tensors,
       HipdnnConvolutionAlgorithmPolicy policy)
      : operation_(std::move(operation)), tensors_(std::move(tensors)),
        policy_(policy),
        setup_capability_(structural_capability(operation_, tensors_)) {
    try {
      require_valid_hipdnn_adapter_contract(setup_capability_,
                                            "hipDNN convolution plan");
      if (!setup_capability_.supported) {
        return;
      }
      if (!setup(hipdnnCreate(&handle_), "hipdnnCreate") ||
          !setup(hipdnnCreateTensorDescriptor(&x_descriptor_),
                 "hipdnnCreateTensorDescriptor(X)") ||
          !setup(hipdnnCreateFilterDescriptor(&w_descriptor_),
                 "hipdnnCreateFilterDescriptor(W)") ||
          !setup(hipdnnCreateTensorDescriptor(&y_descriptor_),
                 "hipdnnCreateTensorDescriptor(Y)") ||
          !setup(hipdnnCreateConvolutionDescriptor(&convolution_descriptor_),
                 "hipdnnCreateConvolutionDescriptor")) {
        return;
      }
      if (!set_tensor(x_descriptor_, tensors_[0], "X") ||
          !set_filter(w_descriptor_, tensors_[1]) ||
          !set_tensor(y_descriptor_, tensors_[2], "Y") || !set_convolution()) {
        return;
      }
      if (operation_.kind == HipdnnConvolutionKind::kConvBiasRelu) {
        const std::array<ReferenceTensor, 2> activation_tensors = {tensors_[2],
                                                                   tensors_[2]};
        const std::vector<ReferenceTensor> activation_layout =
            hipdnn_activation_descriptor_tensors(activation_tensors);
        if (!setup(hipdnnCreateTensorDescriptor(&bias_descriptor_),
                   "hipdnnCreateTensorDescriptor(bias)") ||
            !set_tensor(bias_descriptor_, tensors_[3], "bias") ||
            !setup(hipdnnCreateOpTensorDescriptor(&bias_add_descriptor_),
                   "hipdnnCreateOpTensorDescriptor(bias add)") ||
            !setup(hipdnnSetOpTensorDescriptor(
                       bias_add_descriptor_, HIPDNN_OP_TENSOR_ADD,
                       HIPDNN_DATA_FLOAT, HIPDNN_PROPAGATE_NAN),
                   "hipdnnSetOpTensorDescriptor(bias add)") ||
            !setup(hipdnnCreateTensorDescriptor(&activation_tensor_descriptor_),
                   "hipdnnCreateTensorDescriptor(activation tensor)") ||
            !set_tensor(activation_tensor_descriptor_,
                        activation_layout.front(), "activation tensor") ||
            !setup(hipdnnCreateActivationDescriptor(&activation_descriptor_),
                   "hipdnnCreateActivationDescriptor") ||
            !setup(hipdnnSetActivationDescriptor(activation_descriptor_,
                                                 HIPDNN_ACTIVATION_RELU,
                                                 HIPDNN_PROPAGATE_NAN, 0.0),
                   "hipdnnSetActivationDescriptor(ReLU)")) {
          return;
        }
      }
      query_algorithms();
      if (!setup_capability_.supported) {
        return;
      }
      workspace_size_ = algorithm_workspace_size_;
      if (operation_.kind == HipdnnConvolutionKind::kConvBiasRelu) {
        intermediate_offset_ = align_up(algorithm_workspace_size_, 256);
        const std::size_t intermediate_bytes = tensor_bytes(tensors_[2]);
        if (intermediate_offset_ >
            std::numeric_limits<std::size_t>::max() - intermediate_bytes) {
          throw std::overflow_error(
              "ConvBiasRelu reference workspace size overflows");
        }
        biased_offset_ =
            align_up(intermediate_offset_ + intermediate_bytes, 256);
        if (biased_offset_ >
            std::numeric_limits<std::size_t>::max() - intermediate_bytes) {
          throw std::overflow_error(
              "ConvBiasRelu reference workspace size overflows");
        }
        workspace_size_ = biased_offset_ + intermediate_bytes;
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  [[nodiscard]] HipdnnCapability capability() const {
    return setup_capability_;
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return workspace_size_;
  }

  [[nodiscard]] HipdnnConvolutionAlgorithmPolicy policy() const noexcept {
    return policy_;
  }

  [[nodiscard]] std::optional<int> selected_algorithm() const noexcept {
    if (!selected_algorithm_.has_value()) {
      return std::nullopt;
    }
    return algorithms_[*selected_algorithm_].value;
  }

  [[nodiscard]] std::string_view selected_algorithm_name() const noexcept {
    const std::optional<int> algorithm = selected_algorithm();
    return algorithm.has_value()
               ? hipdnn_convolution_algorithm_name(operation_.kind, *algorithm)
               : std::string_view{};
  }

  HipdnnCapability probe_execute(std::span<const flagdnnBinding_t> bindings,
                                 void *workspace, std::size_t workspace_size,
                                 flagdnnStream_t stream) {
    if (!setup_capability_.supported) {
      return setup_capability_;
    }
    if (selected_algorithm_.has_value()) {
      throw std::logic_error(
          "hipDNN convolution probe requires an unselected candidate");
    }
    validate_workspace(workspace, workspace_size);
    check_hipdnn(
        hipdnnSetStream(handle_, reinterpret_cast<hipdnnStream_t>(stream)),
        "hipdnnSetStream");

    std::string last_reason = std::move(last_rejection_);
    for (std::size_t index = next_algorithm_index_; index < algorithms_.size();
         ++index) {
      next_algorithm_index_ = index + 1;
      const hipdnnStatus_t convolution_status =
          launch_convolution(index, bindings, workspace);
      if (convolution_status != HIPDNN_STATUS_SUCCESS) {
        if (!is_execute_capability_status(convolution_status)) {
          check_hipdnn(convolution_status,
                       "hipDNN convolution actual execute gate");
        }
        last_reason = "algorithm " +
                      std::string(hipdnn_convolution_algorithm_name(
                          operation_.kind, algorithms_[index].value)) +
                      " returned " + status_name(convolution_status);
        continue;
      }

      const hipdnnStatus_t sequence_status =
          finish_conv_bias_relu(bindings, workspace);
      if (sequence_status != HIPDNN_STATUS_SUCCESS) {
        const hipError_t drain_status =
            hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream));
        if (drain_status != hipSuccess &&
            !is_hip_capability_status(drain_status)) {
          throw std::runtime_error(
              std::string("hipDNN convolution probe synchronization failed: ") +
              describe_hip_error(drain_status));
        }
        if (is_execute_capability_status(sequence_status)) {
          return HipdnnCapability::vendor_unsupported(
              "hipDNN ConvBiasRelu primitive sequence returned " +
              status_name(sequence_status));
        }
        check_hipdnn(sequence_status,
                     "hipDNN ConvBiasRelu actual execute gate");
      }

      const hipError_t synchronize_status =
          hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream));
      if (synchronize_status != hipSuccess) {
        const std::string reason =
            std::string("actual execute synchronization returned ") +
            describe_hip_error(synchronize_status);
        if (is_hip_capability_status(synchronize_status)) {
          return HipdnnCapability::vendor_unsupported(reason);
        }
        throw std::runtime_error("hipDNN convolution " + reason);
      }
      selected_algorithm_ = index;
      return {};
    }
    return HipdnnCapability::vendor_unsupported(
        "all hipDNN convolution algorithms were unavailable or rejected" +
        (last_reason.empty() ? std::string() : ": " + last_reason));
  }

  void reject_selected_algorithm(std::string reason) {
    if (policy_ != HipdnnConvolutionAlgorithmPolicy::kPerformance) {
      throw std::logic_error(
          "correctness-oracle convolution algorithm cannot be rejected");
    }
    if (!selected_algorithm_.has_value()) {
      throw std::logic_error(
          "hipDNN convolution rejection requires a selected candidate");
    }
    if (reason.empty()) {
      throw std::invalid_argument(
          "hipDNN convolution candidate rejection needs a reason");
    }
    last_rejection_ =
        "algorithm " + std::string(selected_algorithm_name()) +
        " failed the correctness-oracle comparison: " + std::move(reason);
    selected_algorithm_.reset();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) {
    if (!setup_capability_.supported) {
      throw std::logic_error(
          "cannot execute an unavailable hipDNN convolution plan");
    }
    if (!selected_algorithm_.has_value()) {
      throw std::logic_error(
          "hipDNN convolution execute requires a successful probe_execute");
    }
    validate_workspace(workspace, workspace_size);
    check_hipdnn(
        hipdnnSetStream(handle_, reinterpret_cast<hipdnnStream_t>(stream)),
        "hipdnnSetStream");
    check_hipdnn(launch_convolution(*selected_algorithm_, bindings, workspace),
                 "hipDNN convolution execute");
    check_hipdnn(finish_conv_bias_relu(bindings, workspace),
                 "hipDNN ConvBiasRelu primitive sequence");
  }

private:
  struct Algorithm {
    int value = 0;
    std::size_t workspace_size = 0;
  };

  bool setup(hipdnnStatus_t status, std::string_view stage) {
    if (status == HIPDNN_STATUS_SUCCESS) {
      return true;
    }
    if (is_setup_capability_status(status)) {
      setup_capability_ = HipdnnCapability::vendor_unsupported(
          std::string(stage) + " returned " + status_name(status));
      return false;
    }
    check_hipdnn(status, stage);
    return false;
  }

  bool set_tensor(hipdnnTensorDescriptor_t descriptor,
                  const ReferenceTensor &tensor, std::string_view role) {
    const std::vector<int> dimensions =
        checked_ints(tensor.dimensions, "tensor dimensions", false);
    const std::vector<int> strides =
        checked_ints(tensor.strides, "tensor strides", false);
    return setup(
        hipdnnSetTensorNdDescriptor(descriptor,
                                    hipdnn_data_type(tensor.data_type),
                                    static_cast<int>(dimensions.size()),
                                    dimensions.data(), strides.data()),
        std::string("hipdnnSetTensorNdDescriptor(") + std::string(role) + ")");
  }

  bool set_filter(hipdnnFilterDescriptor_t descriptor,
                  const ReferenceTensor &tensor) {
    const std::vector<int> dimensions =
        checked_ints(tensor.dimensions, "filter dimensions", false);
    const std::optional<hipdnnTensorFormat_t> format = canonical_format(tensor);
    if (!format.has_value()) {
      setup_capability_ = HipdnnCapability::vendor_unsupported(
          "hipDNN filter layout is not canonical");
      return false;
    }
    return setup(hipdnnSetFilterNdDescriptor(
                     descriptor, hipdnn_data_type(tensor.data_type), *format,
                     static_cast<int>(dimensions.size()), dimensions.data()),
                 "hipdnnSetFilterNdDescriptor(W)");
  }

  bool set_convolution() {
    const std::vector<int> padding =
        checked_ints(operation_.padding, "convolution padding", true);
    const std::vector<int> stride =
        checked_ints(operation_.stride, "convolution stride", false);
    const std::vector<int> dilation =
        checked_ints(operation_.dilation, "convolution dilation", false);
    if (!setup(hipdnnSetConvolutionNdDescriptor(
                   convolution_descriptor_, static_cast<int>(padding.size()),
                   padding.data(), stride.data(), dilation.data(),
                   hipdnn_convolution_mode(operation_.mode), HIPDNN_DATA_FLOAT),
               "hipdnnSetConvolutionNdDescriptor")) {
      return false;
    }
    return setup(
        hipdnnSetConvolutionGroupCount(convolution_descriptor_,
                                       static_cast<int>(operation_.groups)),
        "hipdnnSetConvolutionGroupCount");
  }

  void query_algorithms() {
    std::string last_rejection;
    std::vector<int> heuristic_values;
    if (policy_ == HipdnnConvolutionAlgorithmPolicy::kPerformance) {
      const int count = convolution_algorithm_count(operation_.kind);
      int returned = 0;
      hipdnnStatus_t heuristic_status = HIPDNN_STATUS_NOT_SUPPORTED;
      const auto collect = [&](const auto &performance,
                               std::string_view stage) {
        if (returned < 0 || returned > count) {
          throw std::runtime_error(std::string(stage) +
                                   " returned an invalid count");
        }
        for (int index = 0; index < returned; ++index) {
          const auto &candidate = performance[static_cast<std::size_t>(index)];
          if (candidate.status == HIPDNN_STATUS_SUCCESS) {
            heuristic_values.push_back(static_cast<int>(candidate.algo));
          } else if (!hipdnn_convolution_heuristic_allows_enum_fallback(
                         candidate.status)) {
            check_hipdnn(candidate.status,
                         std::string(stage) + " candidate status");
          }
        }
      };

      if (operation_.kind == HipdnnConvolutionKind::kDgrad) {
        std::vector<hipdnnConvolutionBwdDataAlgoPerf_t> performance(
            static_cast<std::size_t>(count));
        heuristic_status = hipdnnGetConvolutionBackwardDataAlgorithm_v7(
            handle_, w_descriptor_, y_descriptor_, convolution_descriptor_,
            x_descriptor_, count, &returned, performance.data());
        if (heuristic_status == HIPDNN_STATUS_SUCCESS) {
          collect(performance, "hipDNN v7 dgrad heuristic");
        }
      } else if (operation_.kind == HipdnnConvolutionKind::kWgrad) {
        std::vector<hipdnnConvolutionBwdFilterAlgoPerf_t> performance(
            static_cast<std::size_t>(count));
        heuristic_status = hipdnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle_, x_descriptor_, y_descriptor_, convolution_descriptor_,
            w_descriptor_, count, &returned, performance.data());
        if (heuristic_status == HIPDNN_STATUS_SUCCESS) {
          collect(performance, "hipDNN v7 wgrad heuristic");
        }
      } else {
        std::vector<hipdnnConvolutionFwdAlgoPerf_t> performance(
            static_cast<std::size_t>(count));
        heuristic_status = hipdnnGetConvolutionForwardAlgorithm_v7(
            handle_, x_descriptor_, w_descriptor_, convolution_descriptor_,
            y_descriptor_, count, &returned, performance.data());
        if (heuristic_status == HIPDNN_STATUS_SUCCESS) {
          collect(performance, "hipDNN v7 fprop heuristic");
        }
      }

      if (heuristic_status != HIPDNN_STATUS_SUCCESS) {
        if (!hipdnn_convolution_heuristic_allows_enum_fallback(
                heuristic_status)) {
          check_hipdnn(heuristic_status,
                       "hipDNN v7 convolution heuristic query");
        }
        heuristic_values.clear();
      }
    }

    const std::vector<int> ordered_values = hipdnn_convolution_algorithm_order(
        operation_.kind, policy_, heuristic_values);

    for (int value : ordered_values) {
      std::size_t workspace = 0;
      hipdnnStatus_t status = HIPDNN_STATUS_NOT_SUPPORTED;
      if (operation_.kind == HipdnnConvolutionKind::kDgrad) {
        status = hipdnnGetConvolutionBackwardDataWorkspaceSize(
            handle_, w_descriptor_, y_descriptor_, convolution_descriptor_,
            x_descriptor_, static_cast<hipdnnConvolutionBwdDataAlgo_t>(value),
            &workspace);
      } else if (operation_.kind == HipdnnConvolutionKind::kWgrad) {
        status = hipdnnGetConvolutionBackwardFilterWorkspaceSize(
            handle_, x_descriptor_, y_descriptor_, convolution_descriptor_,
            w_descriptor_, static_cast<hipdnnConvolutionBwdFilterAlgo_t>(value),
            &workspace);
      } else {
        status = hipdnnGetConvolutionForwardWorkspaceSize(
            handle_, x_descriptor_, w_descriptor_, convolution_descriptor_,
            y_descriptor_, static_cast<hipdnnConvolutionFwdAlgo_t>(value),
            &workspace);
      }
      if (status == HIPDNN_STATUS_SUCCESS) {
        algorithms_.push_back({value, workspace});
        algorithm_workspace_size_ =
            std::max(algorithm_workspace_size_, workspace);
      } else if (is_setup_capability_status(status)) {
        last_rejection = "algorithm " +
                         std::string(hipdnn_convolution_algorithm_name(
                             operation_.kind, value)) +
                         " workspace query returned " + status_name(status);
      } else {
        check_hipdnn(status, "hipDNN convolution workspace query");
      }
    }
    if (algorithms_.empty()) {
      setup_capability_ = HipdnnCapability::vendor_unsupported(
          "hipDNN exposes no executable convolution algorithm for policy " +
          std::string(hipdnn_convolution_algorithm_policy_name(policy_)) +
          (last_rejection.empty() ? std::string()
                                  : ": " + std::move(last_rejection)));
    }
  }

  hipdnnStatus_t launch_convolution(std::size_t algorithm_index,
                                    std::span<const flagdnnBinding_t> bindings,
                                    void *workspace) {
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    const Algorithm &algorithm = algorithms_.at(algorithm_index);
    void *x = binding_pointer(bindings, tensors_[0]);
    void *w = binding_pointer(bindings, tensors_[1]);
    void *y = binding_pointer(bindings, tensors_[2]);
    if (operation_.kind == HipdnnConvolutionKind::kConvBiasRelu) {
      y = static_cast<void *>(static_cast<std::uint8_t *>(workspace) +
                              intermediate_offset_);
    }
    switch (operation_.kind) {
    case HipdnnConvolutionKind::kFprop:
    case HipdnnConvolutionKind::kConvBiasRelu:
      return hipdnnConvolutionForward(
          handle_, &one, x_descriptor_, x, w_descriptor_, w,
          convolution_descriptor_,
          static_cast<hipdnnConvolutionFwdAlgo_t>(algorithm.value), workspace,
          algorithm.workspace_size, &zero, y_descriptor_, y);
    case HipdnnConvolutionKind::kDgrad:
      return hipdnnConvolutionBackwardData(
          handle_, &one, w_descriptor_, w, y_descriptor_, y,
          convolution_descriptor_,
          static_cast<hipdnnConvolutionBwdDataAlgo_t>(algorithm.value),
          workspace, algorithm.workspace_size, &zero, x_descriptor_, x);
    case HipdnnConvolutionKind::kWgrad:
      return hipdnnConvolutionBackwardFilter(
          handle_, &one, x_descriptor_, x, y_descriptor_, y,
          convolution_descriptor_,
          static_cast<hipdnnConvolutionBwdFilterAlgo_t>(algorithm.value),
          workspace, algorithm.workspace_size, &zero, w_descriptor_, w);
    case HipdnnConvolutionKind::kUnavailable:
      break;
    }
    return HIPDNN_STATUS_NOT_SUPPORTED;
  }

  hipdnnStatus_t
  finish_conv_bias_relu(std::span<const flagdnnBinding_t> bindings,
                        void *workspace) {
    if (operation_.kind != HipdnnConvolutionKind::kConvBiasRelu) {
      return HIPDNN_STATUS_SUCCESS;
    }
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    void *intermediate = static_cast<void *>(
        static_cast<std::uint8_t *>(workspace) + intermediate_offset_);
    void *biased = static_cast<void *>(static_cast<std::uint8_t *>(workspace) +
                                       biased_offset_);
    void *bias = binding_pointer(bindings, tensors_[3]);
    void *output = binding_pointer(bindings, tensors_[2]);
    hipdnnStatus_t status = hipdnnOpTensor(
        handle_, bias_add_descriptor_, &one, y_descriptor_, intermediate, &one,
        bias_descriptor_, bias, &zero, y_descriptor_, biased);
    if (status != HIPDNN_STATUS_SUCCESS) {
      return status;
    }
    return hipdnnActivationForward(handle_, activation_descriptor_, &one,
                                   activation_tensor_descriptor_, biased, &zero,
                                   activation_tensor_descriptor_, output);
  }

  void validate_workspace(void *workspace, std::size_t workspace_size) const {
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument("hipDNN convolution workspace is too small");
    }
  }

  void cleanup() noexcept {
    if (activation_descriptor_ != nullptr) {
      (void)hipdnnDestroyActivationDescriptor(activation_descriptor_);
      activation_descriptor_ = nullptr;
    }
    if (activation_tensor_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(activation_tensor_descriptor_);
      activation_tensor_descriptor_ = nullptr;
    }
    if (bias_add_descriptor_ != nullptr) {
      (void)hipdnnDestroyOpTensorDescriptor(bias_add_descriptor_);
      bias_add_descriptor_ = nullptr;
    }
    if (bias_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(bias_descriptor_);
      bias_descriptor_ = nullptr;
    }
    if (convolution_descriptor_ != nullptr) {
      (void)hipdnnDestroyConvolutionDescriptor(convolution_descriptor_);
      convolution_descriptor_ = nullptr;
    }
    if (y_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(y_descriptor_);
      y_descriptor_ = nullptr;
    }
    if (w_descriptor_ != nullptr) {
      (void)hipdnnDestroyFilterDescriptor(w_descriptor_);
      w_descriptor_ = nullptr;
    }
    if (x_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(x_descriptor_);
      x_descriptor_ = nullptr;
    }
    if (handle_ != nullptr) {
      (void)hipdnnDestroy(handle_);
      handle_ = nullptr;
    }
  }

  HipdnnConvolutionOperation operation_;
  std::vector<ReferenceTensor> tensors_;
  HipdnnConvolutionAlgorithmPolicy policy_;
  HipdnnCapability setup_capability_;
  hipdnnHandle_t handle_ = nullptr;
  hipdnnTensorDescriptor_t x_descriptor_ = nullptr;
  hipdnnFilterDescriptor_t w_descriptor_ = nullptr;
  hipdnnTensorDescriptor_t y_descriptor_ = nullptr;
  hipdnnTensorDescriptor_t activation_tensor_descriptor_ = nullptr;
  hipdnnTensorDescriptor_t bias_descriptor_ = nullptr;
  hipdnnOpTensorDescriptor_t bias_add_descriptor_ = nullptr;
  hipdnnConvolutionDescriptor_t convolution_descriptor_ = nullptr;
  hipdnnActivationDescriptor_t activation_descriptor_ = nullptr;
  std::vector<Algorithm> algorithms_;
  std::optional<std::size_t> selected_algorithm_;
  std::size_t next_algorithm_index_ = 0;
  std::string last_rejection_;
  std::size_t algorithm_workspace_size_ = 0;
  std::size_t intermediate_offset_ = 0;
  std::size_t biased_offset_ = 0;
  std::size_t workspace_size_ = 0;
};

HipdnnConvolutionPlan::HipdnnConvolutionPlan(
    HipdnnConvolutionOperation operation, std::vector<ReferenceTensor> tensors,
    HipdnnConvolutionAlgorithmPolicy policy)
    : impl_(std::make_unique<Impl>(std::move(operation), std::move(tensors),
                                   policy)) {}

HipdnnConvolutionPlan::~HipdnnConvolutionPlan() = default;
HipdnnConvolutionPlan::HipdnnConvolutionPlan(
    HipdnnConvolutionPlan &&) noexcept = default;
HipdnnConvolutionPlan &
HipdnnConvolutionPlan::operator=(HipdnnConvolutionPlan &&) noexcept = default;

HipdnnCapability HipdnnConvolutionPlan::capability() const {
  return impl_->capability();
}

std::size_t HipdnnConvolutionPlan::workspace_size() const noexcept {
  return impl_->workspace_size();
}

HipdnnConvolutionAlgorithmPolicy
HipdnnConvolutionPlan::policy() const noexcept {
  return impl_->policy();
}

std::optional<int> HipdnnConvolutionPlan::selected_algorithm() const noexcept {
  return impl_->selected_algorithm();
}

std::string_view
HipdnnConvolutionPlan::selected_algorithm_name() const noexcept {
  return impl_->selected_algorithm_name();
}

HipdnnCapability HipdnnConvolutionPlan::probe_execute(
    std::span<const flagdnnBinding_t> bindings, void *workspace,
    std::size_t workspace_size, flagdnnStream_t stream) {
  return impl_->probe_execute(bindings, workspace, workspace_size, stream);
}

void HipdnnConvolutionPlan::reject_selected_algorithm(std::string reason) {
  impl_->reject_selected_algorithm(std::move(reason));
}

void HipdnnConvolutionPlan::execute(std::span<const flagdnnBinding_t> bindings,
                                    void *workspace, std::size_t workspace_size,
                                    flagdnnStream_t stream) {
  impl_->execute(bindings, workspace, workspace_size, stream);
}

} // namespace flagdnn::validation::hygon
