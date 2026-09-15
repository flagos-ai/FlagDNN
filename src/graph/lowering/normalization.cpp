/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "graph/lowering/helpers.hpp"
#include "graph/lowering/lowering.hpp"

namespace flagdnn::native {

LoweredOperation lower_normalization_forward(const OperationSpec& operation,
                                             bool rmsnorm) {
  require_port_count(operation, 3, rmsnorm ? 2 : 3);
  const char* operation_name = rmsnorm ? "rmsnorm" : "layernorm";
  const TensorSpec& x = require_port(operation.inputs, "x", "input");
  const TensorSpec& scale = require_port(operation.inputs, "scale", "input");
  const TensorSpec& bias = require_port(operation.inputs, "bias", "input");
  const TensorSpec& y = require_port(operation.outputs, "y", "output");
  const TensorSpec* mean =
      rmsnorm ? nullptr : &require_port(operation.outputs, "mean", "output");
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
  require_same_data_type(x, scale,
                         "normalization scale data type must match X");
  require_same_data_type(x, bias, "normalization bias data type must match X");
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

  const std::size_t leading = x.dimensions.size() - scale.dimensions.size();
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
      throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                     "normalization scale must describe a contiguous suffix");
    }
    if (normalized_suffix) {
      normalized_elements =
          checked_multiply(normalized_elements, x.dimensions[axis],
                           "normalization extent overflows");
      statistic_dimensions[axis] = 1;
    }
  }
  if (!normalized_suffix && x.dimensions.back() == 1) normalized_suffix = true;
  if (!normalized_suffix || scale.element_count() != normalized_elements ||
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
  return {{{"rows", rows}, {"normalized_elements", normalized_elements}},
          {{"epsilon", epsilon}},
          {}};
}

LoweredOperation lower_batchnorm(const OperationSpec& operation) {
  require_port_count(operation, 5, 5);
  const TensorSpec& x = require_port(operation.inputs, "x", "input");
  const TensorSpec& scale = require_port(operation.inputs, "scale", "input");
  const TensorSpec& bias = require_port(operation.inputs, "bias", "input");
  const TensorSpec& previous_running_mean =
      require_port(operation.inputs, "previous_running_mean", "input");
  const TensorSpec& previous_running_variance =
      require_port(operation.inputs, "previous_running_variance", "input");
  const TensorSpec& y = require_port(operation.outputs, "y", "output");
  const TensorSpec& mean = require_port(operation.outputs, "mean", "output");
  const TensorSpec& inv_variance =
      require_port(operation.outputs, "inv_variance", "output");
  const TensorSpec& next_running_mean =
      require_port(operation.outputs, "next_running_mean", "output");
  const TensorSpec& next_running_variance =
      require_port(operation.outputs, "next_running_variance", "output");

  for (const auto& item :
       std::initializer_list<std::pair<const TensorSpec*, const char*>>{
           {&x, "batchnorm X"},
           {&scale, "batchnorm scale"},
           {&bias, "batchnorm bias"},
           {&previous_running_mean, "batchnorm previous running mean"},
           {&previous_running_variance, "batchnorm previous running variance"},
           {&y, "batchnorm Y"},
           {&mean, "batchnorm mean"},
           {&inv_variance, "batchnorm inverse variance"},
           {&next_running_mean, "batchnorm next running mean"},
           {&next_running_variance, "batchnorm next running variance"}}) {
    require_non_overlapping_tensor(*item.first, item.second);
  }
  require_same_data_type(x, y, "batchnorm X/Y data types must match");
  require_same_data_type(x, scale, "batchnorm scale data type must match X");
  require_same_data_type(x, bias, "batchnorm bias data type must match X");
  require_floating_data_type(x, "batchnorm X/Y must use a floating data type");

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
       {&previous_running_mean, &previous_running_variance, &mean,
        &inv_variance, &next_running_mean, &next_running_variance}) {
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
    spatial = checked_multiply(spatial, x.dimensions[axis],
                               "batchnorm spatial extent overflows");
  }
  (void)checked_multiply(batch, spatial,
                         "batchnorm reduction extent overflows");
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

LoweredOperation lower_batchnorm_inference(const OperationSpec& operation) {
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
  require_non_overlapping_tensor(inv_variance,
                                 "batchnorm inference inverse variance");
  require_non_overlapping_tensor(scale, "batchnorm inference scale");
  require_non_overlapping_tensor(bias, "batchnorm inference bias");
  require_non_overlapping_tensor(y, "batchnorm inference Y");
  require_same_data_type(x, y, "batchnorm inference X/Y data types must match");
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
  for (const TensorSpec* parameter : {&mean, &inv_variance, &scale, &bias}) {
    if (parameter->element_count() != channels) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "batchnorm inference parameter size must match channels");
    }
    if (!parameter->is_contiguous()) {
      throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                     "batchnorm inference parameters must be contiguous");
    }
  }
  std::int64_t spatial = 1;
  for (std::size_t axis = 2; axis < x.dimensions.size(); ++axis) {
    spatial = checked_multiply(spatial, x.dimensions[axis],
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

LoweredOperation lower_genstats(const OperationSpec& operation) {
  require_port_count(operation, 1, 2);
  const auto& x = require_port(operation.inputs, "x", "input");
  require_non_overlapping_tensor(x, "genstats input");
  require_floating_data_type(x, "genstats input must be floating");
  if (x.dimensions.size() < 2 || x.dimensions.size() > 8)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "genstats input rank must be 2..8");
  std::vector<std::int64_t> shape(x.dimensions.size(), 1);
  shape[1] = x.dimensions[1];
  for (const auto name : {"sum", "sq_sum"}) {
    const auto& output = require_port(operation.outputs, name, "output");
    require_non_overlapping_tensor(output, "genstats output");
    if (output.data_type != FLAGDNN_DATA_FLOAT32 || output.dimensions != shape)
      throw ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "genstats output must be FP32 with channel-only dimensions");
  }
  const auto channels = x.dimensions[1];
  return {{{"channels", channels}, {"reduction", x.element_count() / channels}},
          {},
          {}};
}

LoweredOperation lower_extended_normalization(const OperationSpec& operation) {
  const auto& name = operation.custom_operation_name;
  const bool backward = name.ends_with("_backward");
  const bool rms = name == "rmsnorm_backward";
  const bool batch = name.starts_with("batchnorm");
  const bool instance = name.starts_with("instancenorm");
  const bool adaptive = name.starts_with("adalayernorm");
  require_port_count(operation, backward ? (rms ? 4 : 5) : 3, 3);
  const auto& x = require_port(operation.inputs, "x", "input");
  const auto& scale = require_port(operation.inputs, "scale", "input");
  const auto rank = x.dimensions.size();
  if (rank == 0 || rank > 8 ||
      ((batch || instance) && rank < (instance ? 3U : 2U)))
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization rank is invalid");
  for (const auto& port : operation.inputs) {
    require_non_overlapping_tensor(port.tensor, "normalization input");
    require_floating_data_type(port.tensor,
                               "normalization input must be floating");
  }
  for (const auto& port : operation.outputs) {
    require_non_overlapping_tensor(port.tensor, "normalization output");
    require_floating_data_type(port.tensor,
                               "normalization output must be floating");
  }
  if (scale.data_type != x.data_type && scale.data_type != FLAGDNN_DATA_FLOAT32)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization scale must use X type or FP32");
  if (scale.dimensions.empty() || scale.dimensions.size() > rank)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization scale rank is invalid");
  const auto leading = rank - scale.dimensions.size();
  std::vector<std::int64_t> axes, statistics_shape = x.dimensions;
  std::int64_t reduction = 1;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const auto parameter =
        axis < leading ? 1 : scale.dimensions[axis - leading];
    if ((parameter != 1 && parameter != x.dimensions[axis]) ||
        ((batch || instance) && parameter != (axis == 1 ? x.dimensions[1] : 1)))
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "normalization scale shape is invalid");
    const bool reduce = batch      ? axis != 1
                        : instance ? axis >= 2
                                   : parameter != 1 && (!adaptive || axis != 0);
    if (reduce) {
      axes.push_back(static_cast<std::int64_t>(axis));
      statistics_shape[axis] = 1;
      reduction = checked_multiply(reduction, x.dimensions[axis],
                                   "normalization reduction size overflows");
    }
  }
  // A trailing dimension of one still defines a valid normalization group.
  if (axes.empty() && x.dimensions.back() == 1)
    axes.push_back(static_cast<std::int64_t>(rank - 1));
  if (axes.empty())
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization has no reduction axis");
  const auto check_statistic = [&](const TensorSpec& tensor) {
    if (tensor.dimensions != statistics_shape ||
        tensor.data_type != FLAGDNN_DATA_FLOAT32)
      throw ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "normalization statistics must be FP32 with the reduced dimensions");
  };
  const auto& result =
      require_port(operation.outputs, backward ? "dx" : "y", "output");
  require_same_data_type(x, result, "normalization result type must match X");
  if (result.dimensions != x.dimensions)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization result shape must match X");
  if (backward) {
    const auto& dy = require_port(operation.inputs, "dy", "input");
    require_same_data_type(x, dy, "normalization DY type must match X");
    if (dy.dimensions != x.dimensions)
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "normalization DY shape must match X");
    if (!rms) check_statistic(require_port(operation.inputs, "mean", "input"));
    check_statistic(require_port(operation.inputs, "inv_variance", "input"));
    for (const auto role : {"dscale", "dbias"}) {
      const auto& gradient = require_port(operation.outputs, role, "output");
      if (gradient.dimensions != scale.dimensions ||
          (gradient.data_type != FLAGDNN_DATA_FLOAT32 &&
           gradient.data_type != scale.data_type))
        throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                       "normalization affine gradient must match scale shape "
                       "and use FP32 or scale type");
    }
    return {
        {{"groups", x.element_count() / reduction}, {"reduction", reduction}},
        {},
        {{"axes", axes}}};
  }
  const auto& bias = require_port(operation.inputs, "bias", "input");
  require_same_data_type(scale, bias,
                         "normalization bias type must match scale");
  if (bias.dimensions != scale.dimensions)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization bias shape must match scale");
  check_statistic(require_port(operation.outputs, "mean", "output"));
  check_statistic(require_port(operation.outputs, "inv_variance", "output"));
  const auto epsilon = real_attribute(operation, "epsilon");
  if (!std::isfinite(epsilon) || epsilon <= 0.0)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "normalization epsilon must be finite and positive");
  if (integer_attribute(operation, "forward_phase") != 2)
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "normalization currently requires TRAINING phase");
  return {{{"groups", x.element_count() / reduction}, {"reduction", reduction}},
          {{"epsilon", epsilon}},
          {{"axes", axes}}};
}

LoweredOperation lower_bn_finalize(const OperationSpec& operation) {
  const auto running = integer_attribute(operation, "has_running");
  if (running != 0 && running != 1)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "bn_finalize running flag is invalid");
  require_port_count(operation, running ? 6 : 4, running ? 6 : 4);
  const auto& sum = require_port(operation.inputs, "sum", "input");
  if (sum.dimensions.empty() || sum.dimensions.size() > 8)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "bn_finalize rank must be 1..8");
  const auto axis = sum.dimensions.size() == 1 ? 0U : 1U;
  for (std::size_t i = 0; i < sum.dimensions.size(); ++i)
    if (i != axis && sum.dimensions[i] != 1)
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "bn_finalize tensors must have channel-only dimensions");
  for (const auto& port : operation.inputs) {
    require_non_overlapping_tensor(port.tensor, "bn_finalize input");
    require_floating_data_type(port.tensor,
                               "bn_finalize input must be floating");
    const bool weight = port.name == "scale" || port.name == "bias";
    if (port.tensor.dimensions != sum.dimensions ||
        (!weight && port.tensor.data_type != FLAGDNN_DATA_FLOAT32))
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "bn_finalize input shape or statistics type is invalid");
  }
  require_same_data_type(require_port(operation.inputs, "scale", "input"),
                         require_port(operation.inputs, "bias", "input"),
                         "bn_finalize scale/bias types must match");
  for (const auto& port : operation.outputs) {
    require_non_overlapping_tensor(port.tensor, "bn_finalize output");
    if (port.tensor.dimensions != sum.dimensions ||
        port.tensor.data_type != FLAGDNN_DATA_FLOAT32)
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "bn_finalize outputs require FP32 channel statistics");
  }
  const auto count = real_attribute(operation, "accum_count"),
             epsilon = real_attribute(operation, "epsilon"),
             momentum = real_attribute(operation, "momentum");
  if (!std::isfinite(count) || count < 1.0 || count != std::floor(count) ||
      count > 9007199254740992.0)
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "bn_finalize accumulation count must be a positive exact integer");
  if (!std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(momentum) ||
      momentum < 0 || momentum > 1)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "bn_finalize epsilon or momentum is invalid");
  return {
      {{"has_running", running}, {"channels", sum.element_count()}},
      {{"epsilon", epsilon}, {"accum_count", count}, {"momentum", momentum}},
      {}};
}

}  // namespace flagdnn::native
