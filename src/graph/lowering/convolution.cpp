/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/lowering/lowering.hpp"

#include "graph/lowering/helpers.hpp"

#include <cstdint>
#include <vector>

namespace flagdnn::native {
namespace {

std::int64_t convolution_output_dimension(std::int64_t input,
                                          std::int64_t filter,
                                          std::int64_t pre_padding,
                                          std::int64_t post_padding,
                                          std::int64_t stride,
                                          std::int64_t dilation) {
  const std::int64_t effective = checked_add(
      checked_multiply(filter - 1, dilation, "convolution extent overflows"),
      1,
      "convolution extent overflows");
  const std::int64_t padded = checked_add(
      checked_add(input, pre_padding, "convolution padding overflows"),
      post_padding,
      "convolution padding overflows");
  if (padded < effective) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution filter is larger than padded input");
  }
  return (padded - effective) / stride + 1;
}

}  // namespace

LoweredOperation lower_convolution_fprop(
    const OperationSpec& operation) {
  require_port_count(operation, 2, 1);
  const TensorSpec& input = require_port(operation.inputs, "input", "input");
  const TensorSpec& filter = require_port(operation.inputs, "filter", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(input, "input");
  require_non_overlapping_tensor(filter, "filter");
  require_non_overlapping_tensor(output, "output");
  require_same_data_type(
      input, filter, "convolution input/filter data types must match");
  require_same_data_type(
      input, output, "convolution input/output data types must match");
  require_floating_data_type(
      input, "convolution tensors must use a floating data type");

  const std::int64_t spatial_rank_value =
      integer_attribute(operation, "spatial_rank");
  if (spatial_rank_value < 1 || spatial_rank_value > 3) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "convolution spatial rank must be in [1, 3]");
  }
  const std::size_t spatial_rank =
      static_cast<std::size_t>(spatial_rank_value);
  const std::vector<std::int64_t>& pre_padding =
      integer_array_attribute(operation, "pre_padding");
  const std::vector<std::int64_t>& post_padding =
      integer_array_attribute(operation, "post_padding");
  const std::vector<std::int64_t>& stride =
      integer_array_attribute(operation, "stride");
  const std::vector<std::int64_t>& dilation =
      integer_array_attribute(operation, "dilation");
  const std::int64_t groups = integer_attribute(operation, "groups");
  const std::size_t tensor_rank = spatial_rank + 2;
  if (input.dimensions.size() != tensor_rank ||
      filter.dimensions.size() != tensor_rank ||
      output.dimensions.size() != tensor_rank) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "convolution tensors must have spatial_rank + 2 logical dimensions");
  }
  if (pre_padding.size() != spatial_rank ||
      post_padding.size() != spatial_rank || stride.size() != spatial_rank ||
      dilation.size() != spatial_rank) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "configured convolution has invalid spatial attributes");
  }
  if (groups <= 0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution groups must be positive");
  }
  for (std::size_t index = 0; index < spatial_rank; ++index) {
    if (pre_padding[index] < 0 || post_padding[index] < 0 ||
        stride[index] <= 0 || dilation[index] <= 0) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "convolution padding/stride/dilation is invalid");
    }
  }

  const std::int64_t n = input.dimensions[0];
  const std::int64_t c = input.dimensions[1];
  const std::int64_t k = filter.dimensions[0];
  if (c % groups != 0 || k % groups != 0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution channels must be divisible by groups");
  }
  const std::int64_t channels_per_group = c / groups;
  if (filter.dimensions[1] != channels_per_group) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution filter channels do not match input tensor");
  }

  std::vector<std::int64_t> expected_output = {n, k};
  expected_output.reserve(tensor_rank);
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    expected_output.push_back(convolution_output_dimension(
        input.dimensions[axis + 2],
        filter.dimensions[axis + 2],
        pre_padding[axis],
        post_padding[axis],
        stride[axis],
        dilation[axis]));
  }
  if (output.dimensions != expected_output) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution FProp output shape is incorrect");
  }
  return {{{"spatial_rank", spatial_rank_value},
           {"groups", groups},
           {"n_outputs", output.element_count()}},
          {},
          {{"pre_padding", pre_padding},
           {"post_padding", post_padding},
           {"stride", stride},
           {"dilation", dilation}}};
}

LoweredOperation lower_convolution_backward(
    const OperationSpec& operation,
    bool data_gradient) {
  require_port_count(operation, 2, 1);
  const TensorSpec& loss = require_port(operation.inputs, "dy", "input");
  const TensorSpec& other = require_port(
      operation.inputs, data_gradient ? "w" : "x", "input");
  const TensorSpec& output = require_port(
      operation.outputs, data_gradient ? "dx" : "dw", "output");
  require_non_overlapping_tensor(loss, "convolution loss");
  require_non_overlapping_tensor(
      other, data_gradient ? "convolution filter" : "convolution image");
  require_non_overlapping_tensor(
      output, data_gradient ? "convolution data gradient"
                            : "convolution weight gradient");
  require_same_data_type(
      loss, other, "convolution backward input data types must match");
  require_same_data_type(
      loss, output, "convolution backward output data type must match");
  require_floating_data_type(
      loss, "convolution backward tensors must use a floating data type");

  const std::int64_t spatial_rank_value =
      integer_attribute(operation, "spatial_rank");
  if (spatial_rank_value < 1 || spatial_rank_value > 3) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "convolution spatial rank must be in [1, 3]");
  }
  const std::size_t spatial_rank =
      static_cast<std::size_t>(spatial_rank_value);
  const std::size_t tensor_rank = spatial_rank + 2;
  if (loss.dimensions.size() != tensor_rank ||
      other.dimensions.size() != tensor_rank ||
      output.dimensions.size() != tensor_rank) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "convolution backward tensors must have spatial_rank + 2 dimensions");
  }
  const std::vector<std::int64_t>& pre_padding =
      integer_array_attribute(operation, "pre_padding");
  const std::vector<std::int64_t>& post_padding =
      integer_array_attribute(operation, "post_padding");
  const std::vector<std::int64_t>& stride =
      integer_array_attribute(operation, "stride");
  const std::vector<std::int64_t>& dilation =
      integer_array_attribute(operation, "dilation");
  if (pre_padding.size() != spatial_rank ||
      post_padding.size() != spatial_rank ||
      stride.size() != spatial_rank ||
      dilation.size() != spatial_rank) {
    throw ApiError(
        FLAGDNN_STATUS_INTERNAL_ERROR,
        "configured convolution backward has invalid spatial attributes");
  }
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    if (pre_padding[axis] < 0 || post_padding[axis] < 0 ||
        stride[axis] <= 0 || dilation[axis] <= 0) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "convolution padding/stride/dilation is invalid");
    }
  }
  const std::int64_t groups = integer_attribute(operation, "groups");
  if (groups <= 0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution groups must be positive");
  }
  const std::int64_t convolution_mode =
      integer_attribute(operation, "convolution_mode");
  if (convolution_mode != 0 && convolution_mode != 1) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution mode is invalid");
  }

  const TensorSpec& image = data_gradient ? output : other;
  const TensorSpec& filter = data_gradient ? other : output;
  const std::int64_t n = image.dimensions[0];
  const std::int64_t c = image.dimensions[1];
  const std::int64_t k = filter.dimensions[0];
  if (loss.dimensions[0] != n || loss.dimensions[1] != k) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution loss batch or channel dimension is incorrect");
  }
  if (c % groups != 0 || k % groups != 0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution channels must be divisible by groups");
  }
  if (filter.dimensions[1] != c / groups) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution filter channels do not match image tensor");
  }

  std::vector<std::int64_t> expected_loss = {n, k};
  expected_loss.reserve(tensor_rank);
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    expected_loss.push_back(convolution_output_dimension(
        image.dimensions[axis + 2],
        filter.dimensions[axis + 2],
        pre_padding[axis],
        post_padding[axis],
        stride[axis],
        dilation[axis]));
  }
  if (loss.dimensions != expected_loss) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "convolution backward loss shape is incorrect");
  }

  return {{{"spatial_rank", spatial_rank_value},
           {"groups", groups},
           {"convolution_mode", convolution_mode},
           {"n_outputs", output.element_count()}},
          {},
          {{"pre_padding", pre_padding},
           {"post_padding", post_padding},
           {"stride", stride},
           {"dilation", dilation}}};
}

}  // namespace flagdnn::native
