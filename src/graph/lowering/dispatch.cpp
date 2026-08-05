/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/lowering/lowering.hpp"

#include "error.hpp"

#include <string_view>

namespace flagdnn::native {

std::string_view operation_name(const OperationSpec& specification) {
  if (specification.operation == FLAGDNN_OPERATION_CUSTOM) {
    if (specification.custom_operation_name.empty()) {
      throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                     "custom operation kind is empty");
    }
    return specification.custom_operation_name;
  }
  switch (specification.operation) {
    case FLAGDNN_OPERATION_RELU:
      return "relu";
    case FLAGDNN_OPERATION_ADD:
      return "add";
    case FLAGDNN_OPERATION_REDUCTION:
      switch (reduction_mode(specification)) {
        case FLAGDNN_REDUCTION_ADD:
          return "reduction_sum";
        case FLAGDNN_REDUCTION_AVG:
          return "reduction_avg";
        case FLAGDNN_REDUCTION_MUL:
          return "reduction_mul";
      }
      break;
    case FLAGDNN_OPERATION_CONVOLUTION_FPROP:
      return "convolution_fprop";
    case FLAGDNN_OPERATION_MATMUL:
      return "matmul";
    case FLAGDNN_OPERATION_SDPA:
      return "sdpa";
    case FLAGDNN_OPERATION_SDPA_BACKWARD:
      return "sdpa_backward";
    case FLAGDNN_OPERATION_SDPA_FP8:
      return "sdpa_fp8";
    case FLAGDNN_OPERATION_SDPA_FP8_BACKWARD:
      return "sdpa_fp8_backward";
    case FLAGDNN_OPERATION_POINTWISE:
      return pointwise_operation_name(pointwise_mode(specification));
    case FLAGDNN_OPERATION_CUSTOM:
      break;
  }
  throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR, "unknown operation");
}

LoweredOperation lower_operation(const OperationSpec& operation) {
  if (!operation.configured) {
    throw ApiError(FLAGDNN_STATUS_NOT_INITIALIZED,
                   "operation descriptor is not configured");
  }
  switch (operation.operation) {
    case FLAGDNN_OPERATION_RELU:
      return lower_unary_pointwise(operation);
    case FLAGDNN_OPERATION_ADD:
      return lower_add(operation);
    case FLAGDNN_OPERATION_REDUCTION:
      return lower_reduction(operation);
    case FLAGDNN_OPERATION_CONVOLUTION_FPROP:
      return lower_convolution_fprop(operation);
    case FLAGDNN_OPERATION_MATMUL:
      return lower_matmul(operation);
    case FLAGDNN_OPERATION_SDPA:
      return lower_sdpa(operation);
    case FLAGDNN_OPERATION_SDPA_BACKWARD:
      return lower_sdpa_backward(operation);
    case FLAGDNN_OPERATION_SDPA_FP8:
      return lower_sdpa_fp8(operation);
    case FLAGDNN_OPERATION_SDPA_FP8_BACKWARD:
      return lower_sdpa_fp8_backward(operation);
    case FLAGDNN_OPERATION_POINTWISE:
      return lower_pointwise(operation);
    case FLAGDNN_OPERATION_CUSTOM:
      if (operation.custom_operation_name == "relu") {
        return lower_unary_pointwise(operation);
      }
      if (operation.custom_operation_name == "reshape") {
        return lower_reshape(operation);
      }
      if (operation.custom_operation_name == "transpose") {
        return lower_transpose(operation);
      }
      if (operation.custom_operation_name == "slice") {
        return lower_slice(operation);
      }
      if (operation.custom_operation_name == "layernorm") {
        return lower_normalization_forward(operation, false);
      }
      if (operation.custom_operation_name == "rmsnorm") {
        return lower_normalization_forward(operation, true);
      }
      if (operation.custom_operation_name == "batchnorm") {
        return lower_batchnorm(operation);
      }
      if (operation.custom_operation_name == "batchnorm_inference") {
        return lower_batchnorm_inference(operation);
      }
      if (operation.custom_operation_name == "convolution_dgrad") {
        return lower_convolution_backward(operation, true);
      }
      if (operation.custom_operation_name == "convolution_wgrad") {
        return lower_convolution_backward(operation, false);
      }
      return {};
  }
  throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR, "unknown operation");
}

flagdnnDataType_t operation_compute_data_type(
    const OperationSpec& operation) {
  if (operation.has_compute_data_type) {
    return operation.compute_data_type;
  }
  if (operation.inputs.empty()) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "operation has no input data type");
  }
  return operation.inputs.front().tensor.data_type;
}

}  // namespace flagdnn::native
