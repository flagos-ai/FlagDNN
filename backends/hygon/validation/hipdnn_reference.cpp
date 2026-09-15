/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "hipdnn_reference.hpp"

#include <hip/hip_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string_view>

namespace flagdnn::validation::hygon {
std::string_view hipdnn_status_name(hipdnnStatus_t status) noexcept {
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

bool hipdnn_status_is_capability(hipdnnStatus_t status) noexcept {
  // A reference operation may be skipped only when hipDNN explicitly says
  // that primitive is unsupported.  Missing runtime prerequisites, an
  // architecture mismatch, version skew, initialization failures, and all
  // execution errors indicate a broken validation environment and must fail.
  return status == HIPDNN_STATUS_NOT_SUPPORTED;
}

void require_valid_hipdnn_adapter_contract(const HipdnnCapability &capability,
                                           std::string_view operation) {
  const bool classified_supported =
      capability.classification == HipdnnCapabilityClass::kSupported;
  if (capability.supported != classified_supported ||
      (capability.supported && !capability.reason.empty()) ||
      (!capability.supported && capability.reason.empty())) {
    throw std::invalid_argument(std::string(operation) +
                                " returned a malformed hipDNN capability");
  }
  if (capability.classification ==
      HipdnnCapabilityClass::kInvalidAdapterContract) {
    throw std::invalid_argument(
        std::string(operation) +
        " has an invalid hipDNN adapter contract: " + capability.reason);
  }
}

HipdnnStatusError::HipdnnStatusError(hipdnnStatus_t status,
                                     std::string_view operation)
    : std::runtime_error(std::string(operation) +
                         " failed: " + std::string(hipdnn_status_name(status))),
      status_(status) {}

void check_hipdnn_status(hipdnnStatus_t status, std::string_view operation) {
  if (status != HIPDNN_STATUS_SUCCESS) {
    throw HipdnnStatusError(status, operation);
  }
}

namespace {

std::string dtype_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      return "int32";

    case FLAGDNN_DATA_FP8_E8M0:
      return "fp8_e8m0";
    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_BOOLEAN:
      return "bool";
    case FLAGDNN_DATA_FP8_E4M3:
      return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2:
      return "fp8_e5m2";
  }
  return "unknown";
}

} // namespace

std::string hipdnn_environment() {
  int device = 0;
  hipDeviceProp_t properties{};
  std::ostringstream output;
  output << "header=" << HIPDNN_VERSION << " runtime=" << hipdnnGetVersion();
  if (hipGetDevice(&device) == hipSuccess &&
      hipGetDeviceProperties(&properties, device) == hipSuccess) {
    output << " arch=" << properties.gcnArchName;
  } else {
    output << " arch=unknown";
  }
  return output.str();
}

std::string
describe_reference_tensors(std::span<const ReferenceTensor> tensors) {
  if (tensors.empty()) {
    return "dtype=unknown shape=[]";
  }
  const ReferenceTensor &output = tensors.back();
  std::ostringstream result;
  result << "dtype=" << dtype_name(output.data_type) << " shape=[";
  for (std::size_t index = 0; index < output.dimensions.size(); ++index) {
    if (index != 0) {
      result << 'x';
    }
    result << output.dimensions[index];
  }
  result << "] strides=[";
  for (std::size_t index = 0; index < output.strides.size(); ++index) {
    if (index != 0) {
      result << ',';
    }
    result << output.strides[index];
  }
  result << ']';
  return result.str();
}

} // namespace flagdnn::validation::hygon
