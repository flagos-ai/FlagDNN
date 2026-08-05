/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/lowering/lowering.hpp"

#include "graph/lowering/helpers.hpp"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace flagdnn::native {

LoweredOperation lower_normalization_forward(
    const OperationSpec& operation, bool rmsnorm) {
  require_port_count(operation, 3, rmsnorm ? 2 : 3);
  const char* operation_name = rmsnorm ? "rmsnorm" : "layernorm";
  const TensorSpec& x = require_port(operation.inputs, "x", "input");
  const TensorSpec& scale = require_port(operation.inputs, "scale", "input");
  const TensorSpec& bias = require_port(operation.inputs, "bias", "input");
  const TensorSpec& y = require_port(operation.outputs, "y", "output");
  const TensorSpec* mean =
      rmsnorm
          ? nullptr
          : &require_port(operation.outputs, "mean", "output");
  const TensorSpec& inv_variance =
      require_port(operation.outputs, "inv_variance", "output");

  require_non_overlapping_tensor(x, operation_name);
  require_non_overlapping_tensor(scale, operation_name);
  require_non_overlapping_tensor(bias, operation_name);
  require_non_overlapping_tensor(y, operation_name);
  require_non_overlapping_tensor(inv_variance, operation_name);
  if (mean != nullptr) {
    require_non_overlapping_tensor(*mean, operation_name);
  }
  require_same_data_type(x, y, "normalization X/Y data types must match");
  require_same_data_type(
      x, scale, "normalization scale data type must match X");
  require_same_data_type(
      x, bias, "normalization bias data type must match X");
  require_floating_data_type(
      x, "normalization tensors must use a floating data type");
  if (x.dimensions.empty() || x.dimensions.size() > 8) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "normalization X rank must be in [1, 8]");
  }
  if (y.dimensions != x.dimensions) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization Y shape must match X");
  }
  if (!x.is_contiguous() || !y.is_contiguous()) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "normalization X/Y must be contiguous");
  }
  if (scale.dimensions.empty() ||
      scale.dimensions.size() > x.dimensions.size()) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization scale rank is invalid");
  }

  const std::size_t leading =
      x.dimensions.size() - scale.dimensions.size();
  bool normalized_suffix = false;
  std::int64_t normalized_elements = 1;
  std::vector<std::int64_t> statistic_dimensions = x.dimensions;
  for (std::size_t axis = 0; axis < x.dimensions.size(); ++axis) {
    const std::int64_t scale_dimension =
        axis < leading ? 1 : scale.dimensions[axis - leading];
    if (scale_dimension != 1) {
      if (scale_dimension != x.dimensions[axis]) {
        throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                       "normalization scale shape does not match X");
      }
      normalized_suffix = true;
    } else if (normalized_suffix && x.dimensions[axis] != 1) {
      throw ApiError(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "normalization scale must describe a contiguous suffix");
    }
    if (normalized_suffix) {
      normalized_elements = checked_multiply(
          normalized_elements,
          x.dimensions[axis],
          "normalization extent overflows");
      statistic_dimensions[axis] = 1;
    }
  }
  if (!normalized_suffix ||
      scale.element_count() != normalized_elements ||
      bias.element_count() != normalized_elements) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization scale/bias size is invalid");
  }
  if (!scale.is_contiguous() || !bias.is_contiguous()) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "normalization scale/bias must be contiguous");
  }
  const std::int64_t rows = x.element_count() / normalized_elements;
  for (const TensorSpec* statistic :
       std::initializer_list<const TensorSpec*>{mean, &inv_variance}) {
    if (statistic == nullptr) {
      continue;
    }
    if (statistic->data_type != FLAGDNN_DATA_FLOAT32 ||
        statistic->dimensions != statistic_dimensions) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "normalization statistic metadata is invalid");
    }
    if (!statistic->is_contiguous()) {
      throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                     "normalization statistics must be contiguous");
    }
  }
  const double epsilon = real_attribute(operation, "epsilon");
  if (!std::isfinite(epsilon) || epsilon <= 0.0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization epsilon must be finite and positive");
  }
  if (integer_attribute(operation, "forward_phase") != 2) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "normalization currently supports TRAINING phase only");
  }
  return {{{"rows", rows},
           {"normalized_elements", normalized_elements}},
          {{"epsilon", epsilon}},
          {}};
}

LoweredOperation lower_batchnorm(const OperationSpec& operation) {
  require_port_count(operation, 5, 5);
  const TensorSpec& x = require_port(operation.inputs, "x", "input");
  const TensorSpec& scale = require_port(operation.inputs, "scale", "input");
  const TensorSpec& bias = require_port(operation.inputs, "bias", "input");
  const TensorSpec& previous_running_mean = require_port(
      operation.inputs, "previous_running_mean", "input");
  const TensorSpec& previous_running_variance = require_port(
      operation.inputs, "previous_running_variance", "input");
  const TensorSpec& y = require_port(operation.outputs, "y", "output");
  const TensorSpec& mean = require_port(operation.outputs, "mean", "output");
  const TensorSpec& inv_variance =
      require_port(operation.outputs, "inv_variance", "output");
  const TensorSpec& next_running_mean = require_port(
      operation.outputs, "next_running_mean", "output");
  const TensorSpec& next_running_variance = require_port(
      operation.outputs, "next_running_variance", "output");

  for (const auto& item : std::initializer_list<
           std::pair<const TensorSpec*, const char*>>{
           {&x, "batchnorm X"},
           {&scale, "batchnorm scale"},
           {&bias, "batchnorm bias"},
           {&previous_running_mean, "batchnorm previous running mean"},
           {&previous_running_variance,
            "batchnorm previous running variance"},
           {&y, "batchnorm Y"},
           {&mean, "batchnorm mean"},
           {&inv_variance, "batchnorm inverse variance"},
           {&next_running_mean, "batchnorm next running mean"},
           {&next_running_variance,
            "batchnorm next running variance"}}) {
    require_non_overlapping_tensor(*item.first, item.second);
  }
  require_same_data_type(x, y, "batchnorm X/Y data types must match");
  require_same_data_type(
      x, scale, "batchnorm scale data type must match X");
  require_same_data_type(x, bias, "batchnorm bias data type must match X");
  require_floating_data_type(
      x, "batchnorm X/Y must use a floating data type");

  if (x.dimensions.size() < 2 || x.dimensions.size() > 8) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "batchnorm X rank must be in [2, 8]");
  }
  if (y.dimensions != x.dimensions) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "batchnorm Y shape must match X");
  }
  const std::int64_t batch = x.dimensions[0];
  const std::int64_t channels = x.dimensions[1];
  for (const TensorSpec* parameter : {&scale, &bias}) {
    if (parameter->element_count() != channels) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "batchnorm scale/bias size must match channels");
    }
    if (!parameter->is_contiguous()) {
      throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                     "batchnorm scale/bias must be contiguous");
    }
  }
  for (const TensorSpec* statistic :
       {&previous_running_mean,
        &previous_running_variance,
        &mean,
        &inv_variance,
        &next_running_mean,
        &next_running_variance}) {
    if (statistic->data_type != FLAGDNN_DATA_FLOAT32) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "batchnorm statistics must use float32");
    }
    if (statistic->element_count() != channels) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "batchnorm statistic size must match channels");
    }
    if (!statistic->is_contiguous()) {
      throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                     "batchnorm statistics must be contiguous");
    }
  }

  const double epsilon = real_attribute(operation, "epsilon");
  const double momentum = real_attribute(operation, "momentum");
  if (!std::isfinite(epsilon) || epsilon <= 0.0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "batchnorm epsilon must be finite and positive");
  }
  if (!std::isfinite(momentum) || momentum < 0.0 || momentum > 1.0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "batchnorm momentum must be in [0, 1]");
  }

  std::int64_t spatial = 1;
  for (std::size_t axis = 2; axis < x.dimensions.size(); ++axis) {
    spatial = checked_multiply(
        spatial, x.dimensions[axis], "batchnorm spatial extent overflows");
  }
  (void)checked_multiply(
      batch, spatial, "batchnorm reduction extent overflows");
  return {{{"n_elements", x.element_count()},
           {"batch", batch},
           {"channels", channels},
           {"spatial", spatial},
           {"rank", static_cast<std::int64_t>(x.dimensions.size())}},
          {{"epsilon", epsilon}, {"momentum", momentum}},
          {{"dimensions", x.dimensions},
           {"x_strides", x.strides},
           {"y_strides", y.strides}}};
}

LoweredOperation lower_batchnorm_inference(
    const OperationSpec& operation) {
  require_port_count(operation, 5, 1);
  const TensorSpec& x = require_port(operation.inputs, "x", "input");
  const TensorSpec& mean = require_port(operation.inputs, "mean", "input");
  const TensorSpec& inv_variance =
      require_port(operation.inputs, "inv_variance", "input");
  const TensorSpec& scale = require_port(operation.inputs, "scale", "input");
  const TensorSpec& bias = require_port(operation.inputs, "bias", "input");
  const TensorSpec& y = require_port(operation.outputs, "y", "output");

  require_non_overlapping_tensor(x, "batchnorm inference X");
  require_non_overlapping_tensor(mean, "batchnorm inference mean");
  require_non_overlapping_tensor(
      inv_variance, "batchnorm inference inverse variance");
  require_non_overlapping_tensor(scale, "batchnorm inference scale");
  require_non_overlapping_tensor(bias, "batchnorm inference bias");
  require_non_overlapping_tensor(y, "batchnorm inference Y");
  require_same_data_type(
      x, y, "batchnorm inference X/Y data types must match");
  require_floating_data_type(
      x, "batchnorm inference X/Y must use a floating data type");
  require_floating_data_type(
      mean, "batchnorm inference mean must use a floating data type");
  require_floating_data_type(
      inv_variance,
      "batchnorm inference inverse variance must use a floating data type");
  require_floating_data_type(
      scale, "batchnorm inference scale must use a floating data type");
  require_floating_data_type(
      bias, "batchnorm inference bias must use a floating data type");

  if (x.dimensions.size() < 2 || x.dimensions.size() > 8) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "batchnorm inference X rank must be in [2, 8]");
  }
  if (y.dimensions != x.dimensions) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "batchnorm inference Y shape must match X");
  }
  const std::int64_t channels = x.dimensions[1];
  for (const TensorSpec* parameter :
       {&mean, &inv_variance, &scale, &bias}) {
    if (parameter->element_count() != channels) {
      throw ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "batchnorm inference parameter size must match channels");
    }
    if (!parameter->is_contiguous()) {
      throw ApiError(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "batchnorm inference parameters must be contiguous");
    }
  }
  std::int64_t spatial = 1;
  for (std::size_t axis = 2; axis < x.dimensions.size(); ++axis) {
    spatial = checked_multiply(
        spatial,
        x.dimensions[axis],
        "batchnorm inference spatial extent overflows");
  }
  return {{{"n_elements", x.element_count()},
           {"channels", channels},
           {"spatial", spatial},
           {"rank", static_cast<std::int64_t>(x.dimensions.size())}},
          {},
          {{"dimensions", x.dimensions},
           {"x_strides", x.strides},
           {"y_strides", y.strides}}};
}

}  // namespace flagdnn::native
